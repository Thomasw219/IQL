from comet_ml import Experiment

import os
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.distributions.normal as Normal
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import multiprocessing as mp

from multiprocessing import Pool
from functools import partial

from utils import EpisodeDataBuffer, TransitionDataBuffer
from dynamics import gather_data, gather_data_helper
from environments import ToyEnv, env_random_action
from variational_dynamics import HighLevelStabilityRecurrentModel, HighLevelStabilityRecurrentModelICNNLyapunov, VAE
from gcrl import IQLTrainer

def train_variational_dynamics(model, optimizer, dataloader, total, epochs, start_epoch, length, experiment):
    state_decoder_norms = []
    action_decoder_norms = []
    fixed_point_norms = []

    for epoch in range(start_epoch, start_epoch + epochs):
        total_loss = 0
        total_kl_loss = 0
        total_recon_state_loss = 0
        total_recon_reward_loss = 0
        total_recon_action_loss = 0

        for input_data, output_target in iter(dataloader):
            batch_size = input_data.size()[0]
            losses = model.find_loss(input_data, length)

            loss = losses[0]
            kl_loss = losses[1]
            recon_state_loss = losses[2]
            recon_reward_loss = losses[3]
            recon_action_loss = losses[4]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if model.stable:
                fixed_point_norms.append(torch.linalg.norm(model.fixed_point_module.network[0].weight.grad))
            state_decoder_norms.append(torch.linalg.norm(model.decoder_state_mean.network[0].weight.grad))
            action_decoder_norms.append(torch.linalg.norm(model.decoder_action_mean.network[0].weight.grad))

            total_loss += loss.to(torch.device("cpu")).item() * batch_size / total
            total_kl_loss += kl_loss.to(torch.device("cpu")).item() * batch_size / total
            total_recon_state_loss += recon_state_loss.to(torch.device("cpu")).item() * batch_size / total
            total_recon_reward_loss += recon_reward_loss.to(torch.device("cpu")).item() * batch_size / total
            total_recon_action_loss += recon_action_loss.to(torch.device("cpu")).item() * batch_size / total

        print("Epoch: {}".format(epoch))
        print("Average train loss: {}".format(total_loss))
        if experiment is not None:
            experiment.log_metric("Total Loss", total_loss, epoch=epoch)
            experiment.log_metric("KL Loss", total_kl_loss, epoch=epoch)
            experiment.log_metric("State Reconstruction Loss", total_recon_state_loss, epoch=epoch)
            experiment.log_metric("Reward Reconstruction Loss", total_recon_reward_loss, epoch=epoch)
            experiment.log_metric("Action Reconstruction Loss", total_recon_action_loss, epoch=epoch)
            experiment.log_metric("State Decoder Gradient Norm", torch.mean(torch.stack(state_decoder_norms)), epoch=epoch)
            experiment.log_metric("Action Decoder Gradient Norm", torch.mean(torch.stack(action_decoder_norms)), epoch=epoch)
            if model.stable:
                experiment.log_metric("Fixed Point Embedding Gradient Norm", torch.mean(torch.stack(fixed_point_norms)), epoch=epoch)

def save_local_and_experiment(fig_name, experiment, fig):
#    plt.savefig(fig_name)
    if experiment is not None:
        experiment.log_figure(figure=fig, figure_name=fig_name)
    plt.close(fig)

def make_pertubations(length, p=0.05, scale=1.0):
    pertubations = []
    n = 0
    for i in range(length):
        if np.random.uniform() < 0.05 and n < 2:
            pertubations.append(np.random.normal(scale=[scale, scale]))
            n += 1
        else:
            pertubations.append(np.array([0, 0]))
    return np.array(pertubations).reshape((1, length, 2))

def make_bias(length, scale=0.10):
    bias = np.random.normal(scale=[scale, scale])
    return np.tile(bias, (1, length, 1))

def constant_skill_model_env_rollout_figures(model, env, skill_dim, sequence_length, n_trajs, epochs, n_skills, experiment, device):
    initial_states = env.get_random_state(n=n_trajs)
    torch_initial_states = torch.tensor(initial_states, dtype=torch.float32, device=device)

    initial_means, initial_stds = model.get_prior_state(torch_initial_states)
    initial_dists = Normal.Normal(initial_means, initial_stds)

    for i in range(n_skills):
        state = env.get_random_state()
        torch_state = torch.tensor(state, dtype=torch.float32)
        mean, std = model.get_prior_state(torch_state)
        torch_skill = torch.normal(mean, std)
        skill = torch_skill.cpu().detach().numpy().flatten()
        fixed_point = model.get_fixed_point(torch_skill).cpu().detach().numpy().flatten()

        fig = plt.figure()
        norm = mpl.colors.LogNorm(vmin=1, vmax=sequence_length)
        colors = cm.rainbow(norm(np.array([k for k in range(1, sequence_length + 1)])))

        log_probs = initial_dists.log_prob(torch_skill)
        normalized_log_probs = log_probs - torch.log(std * np.sqrt(2 * np.pi))
        joint_log_probs = torch.sum(normalized_log_probs, dim=1)
        probable_initial_states = torch_initial_states[joint_log_probs > torch.log(torch.tensor(0.1)) * skill_dim]

        for initial_state in probable_initial_states:
            states, _, _ = model.constant_skill_env_rollout(env, sequence_length, skill, initial_state=initial_state)
            x = states[:, 0]
            y = states[:, 1]
            for t in range(sequence_length - 1):
                plt.plot(x[t:t + 2], y[t:t + 2], c=colors[t])

        plt.scatter([fixed_point[0]], [fixed_point[1]], c='k', label='Fixed point for particular skill')
        plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.rainbow),
                label="Time step")

        plt.xlim(-10, 10)
        plt.ylim(-10, 10)
        plt.legend()
        fig_name = "../figures/toy_stable_dynamics_{:04d}_env.png".format(epochs)
        save_local_and_experiment(fig_name, experiment, fig)

        fig = plt.figure()
        norm = mpl.colors.LogNorm(vmin=1, vmax=sequence_length)
        colors = cm.rainbow(norm(np.array([k for k in range(1, sequence_length + 1)])))

        for initial_state in probable_initial_states:
            states, _, _ = model.constant_skill_model_rollout(env, sequence_length, skill, initial_state=initial_state)
            x = states[:, 0]
            y = states[:, 1]
            for t in range(sequence_length - 1):
                plt.plot(x[t:t + 2], y[t:t + 2], c=colors[t])

        plt.scatter([fixed_point[0]], [fixed_point[1]], c='k', label='Fixed point for particular skill')
        plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.rainbow),
                label="Time step")

        plt.xlim(-10, 10)
        plt.ylim(-10, 10)
        plt.legend()
        fig_name = "../figures/toy_stable_dynamics_{:04d}_model.png".format(epochs)
        save_local_and_experiment(fig_name, experiment, fig)

        # Env rollout with pertubations
        fig = plt.figure()
        norm = mpl.colors.LogNorm(vmin=1, vmax=sequence_length)
        colors = cm.rainbow(norm(np.array([k for k in range(1, sequence_length + 1)])))
        for initial_state in probable_initial_states:
            pertubations = make_pertubations(sequence_length)
            states, _, _ = model.constant_skill_env_rollout(env, sequence_length, skill, disturbances=pertubations, initial_state=initial_state)
            x = states[:, 0]
            y = states[:, 1]
            for t in range(sequence_length - 1):
                if np.linalg.norm(pertubations[0, t, :]) == 0:
                    c = colors[t]
                else:
                    c = 'k'
                plt.plot(x[t:t + 2], y[t:t + 2], c=c)

        plt.scatter([fixed_point[0]], [fixed_point[1]], c='k', label='Fixed point for particular skill')
        plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.rainbow),
                label="Time step")

        plt.xlim(-10, 10)
        plt.ylim(-10, 10)
        plt.legend()
        fig_name = "../figures/toy_stable_dynamics_{:04d}_env_pertubations.png".format(epochs)
        save_local_and_experiment(fig_name, experiment, fig)

        # Env rollout with bias
        fig = plt.figure()
        norm = mpl.colors.LogNorm(vmin=1, vmax=sequence_length)
        colors = cm.rainbow(norm(np.array([k for k in range(1, sequence_length + 1)])))
        bias = make_bias(sequence_length)
        for initial_state in probable_initial_states:
            states, _, _ = model.constant_skill_env_rollout(env, sequence_length, skill, disturbances=bias, initial_state=initial_state)
            x = states[:, 0]
            y = states[:, 1]
            for t in range(sequence_length - 1):
                plt.plot(x[t:t + 2], y[t:t + 2], c=colors[t])

        plt.scatter([fixed_point[0]], [fixed_point[1]], c='k', label='Fixed point for particular skill')
        plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.rainbow),
                label="Time step")

        plt.arrow(9, 9, bias[0, 0, 0] * 2, bias[0, 0, 1] * 2, label='Bias', width=0.002, head_width=0.01)

        plt.xlim(-10, 10)
        plt.ylim(-10, 10)
        plt.legend()
        fig_name = "../figures/toy_stable_dynamics_{:04d}_env_bias.png".format(epochs)
        save_local_and_experiment(fig_name, experiment, fig)

def train_data_trajectory_predictions_figures(model, test_traj, epochs, length, experiment, device):
    states, actions, predicted_next_states, predicted_actions, fixed_points = model.model_propagation_visualization(test_traj, length)
    for i in range(states.shape[0]):
        fig = plt.figure()
        plt.scatter(states[i, :, 0], states[i, :, 1], c='b', label='True Trajectory States')
        plt.quiver(states[i, :, 0], states[i, :, 1], actions[i, :, 0], actions[i, :, 1], color='b', label='True Trajectory Actions', angles='xy', scale=1, scale_units='xy', width=0.01)
        colors = cm.rainbow(np.linspace(0, 1, states.shape[1]))
        plt.scatter(predicted_next_states[i, :, 0], predicted_next_states[i, :, 1], color=colors, label='Predicted Next States')
        plt.quiver(states[i, :, 0], states[i, :, 1], predicted_actions[i, :, 0], predicted_actions[i, :, 1], color=colors, label='Predicted Next Actions', angles='xy', scale=1, scale_units='xy', width=0.0075)
        plt.scatter(fixed_points[i, :, 0], fixed_points[i, :, 1], color=colors, label='Fixed Points for Executed Skills', marker='*')
#        plt.xlim(-10, 10)
#        plt.ylim(-10, 10)
        plt.legend()
        fig_name = "../figures/toy_stable_dynamics_{:04d}_test_traj_vis_{:02d}.png".format(epochs, i)

        save_local_and_experiment(fig_name, experiment, fig)

def fixed_points_of_sampled_skills_figures(model, env, skill_dim, epochs, experiment, device, n_samples=2):
    for i in range(n_samples):
        fig = plt.figure()
        np_state = env.get_random_state()
        state = torch.tensor(np_state, dtype=torch.float32).unsqueeze(0)
        mean, std = model.get_prior_state(state)
        skill = np.random.normal(loc=mean.cpu().detach().numpy(), scale=std.cpu().detach().numpy(), size=(250, skill_dim))
        fixed_points = model.get_fixed_point(torch.tensor(skill, device=device, dtype=torch.float32)).cpu().detach().numpy()
        plt.scatter(fixed_points[:, 0], fixed_points[:, 1], c='b')
        plt.scatter([np_state[0]], [np_state[1]], c='r', label='State for Prior')
        plt.xlim(-10, 10)
        plt.ylim(-10, 10)
        plt.legend()
        fig_name = "../figures/toy_stable_dynamics_{:04d}_skill_fixed_points_{:02d}.png".format(epochs, n_samples)
        save_local_and_experiment(fig_name, experiment, fig)

def toy_dynamics_offline_learning_stable_variational_skill_dynamics(cuda_num):
    print("Start")
    state_dim = 2
    action_dim = 2
    embed_dim = 512
    hidden_dim = 512
    skill_dim = 5

    icnn_hidden_dim = 256
    convex_layer_sizes = [state_dim, icnn_hidden_dim, icnn_hidden_dim, 1]
    nonconvex_layer_sizes = [skill_dim, icnn_hidden_dim, icnn_hidden_dim]

    sequence_length = 50

    random_gather_episodes = 5000
    env = ToyEnv()

    batch_size = 512
    fixed_point_lr = 0.000010
    lr = 0.000010
    weight_decay = 1e-5
    stability_param = 0.9

    stable = False
    use_comet = True

    device = torch.device("cuda:{}".format(cuda_num))
    model_params = {"state_dim" : state_dim,
        "action_dim" : action_dim,
        "embed_dim" : embed_dim,
        "hidden_dim" : hidden_dim,
        "skill_dim" : skill_dim,
        "learning_rate" : lr,
        "fixed_point_learning_rate" : fixed_point_lr,
        "stability_param" : stability_param,
        "random_gather_episodes_data": random_gather_episodes,
        "weight_decay" : weight_decay,
        "batch_size" : batch_size,
        "stable" : stable,
        "icnn_convex_layer_sizes" : convex_layer_sizes,
        "icnn_nonconvex_layer_sizes" : nonconvex_layer_sizes}
    model = HighLevelStabilityRecurrentModelICNNLyapunov(**model_params)
    model.to(device)

    fixed_point_params = []
    other_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "fixed_point" in name:
                print(name)
                fixed_point_params.append(param)
            else:
                other_params.append(param)


    model_optimizer = torch.optim.Adam([
            {"params" : fixed_point_params, "lr" : fixed_point_lr},
            {"params" : other_params, "lr" : lr}], weight_decay=weight_decay)
    """
    model_optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    """

    episode_buffer = EpisodeDataBuffer(batch_size, device=device)

    gather_data(env, random_gather_episodes, episode_buffer, env.random_action, sequence_length)

    vis_length = 10
    test_traj = episode_buffer.datasets[0].tensors[0][0:3, 0:vis_length]

    n_skills = 2
    n_trajs = 25

    cycle_epochs = 5
    n_cycles = 250

    if not use_comet:
        experiment = None
    else:
        experiment = Experiment(
                api_key="e1Xmlzbz1cCLgwe0G8m7G58ns",
                project_name="stable-mbrl",
                workspace="thomasw219",
                auto_output_logging="native")
        experiment.log_parameters(model_params)

        if not stable:
            experiment.add_tag("No lyapunov constraint")
#        experiment.add_tag("Stable from beginning")
        experiment.add_tag("Stable from middle")
        experiment.add_tag("Lyapunov in original state space")
        experiment.add_tag("State and reward as input to decoders")
        experiment.add_tag("Post refactor")
        if lr != fixed_point_lr:
            experiment.add_tag("Different learning rate for fixed point module")
        experiment.add_tag("Decrement stability requirement")
#        experiment.add_tag("State conditioned prior")
        experiment.add_tag("Learned ICNN Lyuapunov Function")

    print("Experiment Started", flush=True)

    dataloader = episode_buffer.get_dataloader()
    total = len(episode_buffer)

    length = 50
    for cycle in range(n_cycles):
        with torch.no_grad():
            constant_skill_model_env_rollout_figures(model, env, skill_dim, sequence_length, n_trajs, cycle * cycle_epochs, n_skills, experiment, device)
            train_data_trajectory_predictions_figures(model, test_traj, cycle * cycle_epochs, vis_length, experiment, device)
            fixed_points_of_sampled_skills_figures(model, env, skill_dim, cycle * cycle_epochs, experiment, device)
        if cycle == 50:
            model.stable = True
        elif cycle == 100:
            model.set_gamma(0.7)
        elif cycle == 150:
            model.set_gamma(0.5)
        train_variational_dynamics(model, model_optimizer, dataloader, total, cycle_epochs,
                cycle * cycle_epochs, length, experiment)
        if use_comet:
            torch.save(model.state_dict(), "../models/toy_dynamics_{}.pt".format(experiment.get_name()))

def generate_data(env, episode_len, num_trajectories, action_func, episode_buffer, transition_buffer, processes=8):
    arg_list = [(env, action_func, episode_len) for i in range(num_trajectories)]

    with Pool(processes) as p:
        output = p.starmap(gather_data_helper, arg_list)

    all_states = []
    all_actions = []
    all_rewards = []

    all_next_states = []
    all_next_rewards = []
    for states, actions, rewards, next_states, next_rewards in output:
        all_states.append(states)
        all_actions.append(actions)
        all_rewards.append(rewards)
        all_next_states.append(next_states)
        all_next_rewards.append(next_rewards)

    episode_buffer.add_data(np.stack(all_states),
            np.stack(all_actions),
            np.stack(all_rewards),
            np.stack(all_next_states),
            np.stack(all_next_rewards))

    transition_buffer.add_data(np.concatenate(all_states),
            np.concatenate(all_actions),
            np.concatenate(all_next_states),
            np.concatenate(all_next_rewards))

def train_vae(vae, trajectory_buffer, path, device, args):
    epochs = args["vae"]["epochs"]
    lr = args["vae"]["lr"]
    weight_decay = args["vae"]["weight_decay"]
    train_proportion = args["vae"]["train_proportion"]

    train_loader, test_loader = trajectory_buffer.get_train_test_dataloaders(train_proportion=train_proportion)
    total = len(trajectory_buffer)
    train_total = int(train_proportion * total)
    test_total = total - train_total

    state_dim = args["env"]["state_dim"]

    optimizer = torch.optim.Adam(vae.parameters(), lr=lr, weight_decay=weight_decay)

    best_test_loss = np.inf
    for epoch in range(epochs):
        train_loss_avg = 0
        for input_data, _ in iter(train_loader):
            batch_size = input_data.size()[0]
            states = input_data[:, :state_dim]
            loss, _, _ = vae.get_loss(states)
#            loss *= train_total / batch_size

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_avg += loss.item() * batch_size / train_total

        states = states[:5]
        print(states)
        reconstructions = vae.forward(states)
        print(reconstructions)
        print(torch.abs(states - reconstructions))
        print(torch.square(reconstructions[:, 0]) + torch.square(reconstructions)[:, 1])

        test_loss_avg = 0
        test_recon_avg = 0
        test_kl_avg = 0
        with torch.no_grad():
            for input_data, _ in iter(test_loader):
                batch_size = input_data.size()[0]
                states = input_data[:, :state_dim]
                loss, kl, recon = vae.get_loss(states)

                test_loss_avg += loss.item() * batch_size / test_total
                test_kl_avg += kl.item() * batch_size / test_total
                test_recon_avg += recon.item() * batch_size / test_total

        if args["vae"]["verbose_training"]:
            print("epoch: {}, avg train loss: {}, avg test loss: {}".format(epoch, train_loss_avg, test_loss_avg))
            print("avg kl loss: {}, avg reconstruction loss: {}".format(test_kl_avg, test_recon_avg))
        if test_loss_avg < best_test_loss:
            torch.save(vae.state_dict(), path)

def openai_gym_stable_skill_learning(args):
    env_name = args["env"]["name"]
    num_trajectories = args["env"]["num_trajectories"]
    episode_len = args["env"]["episode_len"]

    device = torch.device(args["device"])

    setting_name = "{}_{}_{}".format(env_name, num_trajectories, episode_len)

    batch_size = args["batch_size"]
    transition_buffer = TransitionDataBuffer(batch_size, device=device)
    episode_buffer = EpisodeDataBuffer(batch_size, device=device)

    transition_path = "../data/{}_transitions.npz".format(setting_name)
    episode_path = "../data/{}_episodes.npz".format(setting_name)

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    args["env"]["state_dim"] = state_dim
    action_dim = env.action_space.shape[0]
    args["env"]["action_dim"] = action_dim

    if os.path.exists(transition_path):
        print("Loading environment data from previous gather")
        transition_buffer.load_data(transition_path)
        episode_buffer.load_data(episode_path)
    else:
        print("Gathering data with random actions")
        random_action_func = partial(env_random_action, env)
        generate_data(env, episode_len, num_trajectories, random_action_func, episode_buffer, transition_buffer)
        transition_buffer.save_data(transition_path)
        episode_buffer.save_data(episode_path)

    vae_path = "../models/{}_vae.pt".format(setting_name)
    vae = VAE(state_dim, args["vae"]["hidden_dim"], args["vae"]["latent_dim"])
    vae.to(device)
    if not os.path.exists(vae_path) or args["vae"]["force_train"]:
        print("Training VAE")
        train_vae(vae, transition_buffer, vae_path, device, args)
    else:
        print("Loading saved VAE")
        vae.load_state_dict(torch.load(vae_path, map_location=device))

def iql_test(cuda_num):
    iql_args = {
        "device" : torch.device("cuda:{}".format(cuda_num)),
        "env_name" : "maze2d-umaze-v1",
        "q_net_args" : {
            "nonlinearity_class" : nn.ReLU,
            "layer_sizes" : [256, 256],
        },
        "q_lr" : 3e-4,
        "q_opt_class" : torch.optim.Adam,
        "v_net_args" : {
            "nonlinearity_class" : nn.ReLU,
            "layer_sizes" : [256, 256],
        },
        "v_lr" : 3e-4,
        "v_opt_class" : torch.optim.Adam,
        "policy_args" : {
            "nonlinearity_class" : nn.ReLU,
            "layer_sizes" : [256, 256],
        },
        "policy_lr" : 3e-4,
        "policy_opt_class" : torch.optim.Adam,
        "gamma" : 0.99,
        "expectile" : 0.9,
        "polyak" : 0.005,
        "beta" : 3.0,
        "batch_size" : 512,
        "n_critic_updates" : int(1e6),
        "n_actor_updates" : int(2e5),
        "save_interval" : 1000,
        "save_dir" : "../models",
        "save_name" : "maze_2d_bigger_intervals",
        "verbose" : True,
        "visualize" : True,
        "vis_points" : 2000,
        "figures_dir" : "../figures",
        "figures_interval" : 10000,
        "eval_interval" : 10000,
        "eval_samples" : 100,
        "joint" : True,
    }
    trainer = IQLTrainer(**iql_args)
    trainer.train_joint()

if __name__ == "__main__":
    mp.set_start_method("spawn")
    cuda_num = 1
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.5, device=cuda_num)

    """
    openai_test_args = {
        "batch_size" : 512,
        "device" : "cuda:{}".format(cuda_num),
        "env" : {
            "name" : "Pendulum-v1",
            "num_trajectories" : 5000,
            "episode_len" : 50,
        },
        "vae" : {
            "hidden_dim" : 128,
            "latent_dim" : 3,
            "epochs" : 250,
            "lr" : 1e-3,
            "weight_decay" : 1e-5,
            "train_proportion" : 0.9,
            "verbose_training" : True,
            "force_train" : False,
        },
    }

    openai_gym_stable_skill_learning(openai_test_args)
    """
    iql_test(cuda_num)
