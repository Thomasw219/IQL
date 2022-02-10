import os
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions.normal as normal
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import d4rl

from copy import deepcopy

class TDNetwork(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.state_dim = kwargs["state_dim"]
        self.action_dim = kwargs["action_dim"]

        self.nonlinearity_class = kwargs["nonlinearity_class"]
        self.layer_sizes = kwargs["layer_sizes"]
        assert len(self.layer_sizes) >= 1

        self.type = kwargs["type"]
        assert self.type == "Q" or self.type == "V"

        if self.type == "Q":
            input_size = self.state_dim + self.action_dim
        else:
            input_size = self.state_dim

        layer_list = [nn.Linear(input_size, self.layer_sizes[0]), self.nonlinearity_class()]
        for i in range(len(self.layer_sizes) - 1):
            layer_list.append(nn.Linear(self.layer_sizes[i], self.layer_sizes[i + 1]))
            layer_list.append(self.nonlinearity_class())
        layer_list.append(nn.Linear(self.layer_sizes[-1], 1))

        for layer in layer_list:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))

        self.network = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.network(x)

class Policy(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.state_dim = kwargs["state_dim"]
        self.action_dim = kwargs["action_dim"]

        self.nonlinearity_class = kwargs["nonlinearity_class"]
        self.layer_sizes = kwargs["layer_sizes"]

        self.state_based_var = kwargs["state_based_var"]
        assert len(self.layer_sizes) >= 1

        layer_list = [nn.Linear(self.state_dim, self.layer_sizes[0]), self.nonlinearity_class()]
        for i in range(len(self.layer_sizes) - 1):
            layer_list.append(nn.Linear(self.layer_sizes[i], self.layer_sizes[i + 1]))
            layer_list.append(self.nonlinearity_class())

        for layer in layer_list:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))

        self.features = nn.Sequential(*layer_list)
        self.means = nn.Linear(self.layer_sizes[-1], self.action_dim)
        nn.init.orthogonal_(self.means.weight, gain=np.sqrt(2))
        if self.state_based_var:
            self.log_vars = nn.Linear(self.layer_sizes[-1], self.action_dim)
            nn.init.orthogonal_(self.log_vars.weight, gain=np.sqrt(2))
        else:
            self.log_var = nn.Parameter(torch.zeros(self.action_dim))

    def forward_dist(self, states):
        features = self.features(states)
        means = torch.tanh(self.means(features))
        if self.state_based_var:
            stds = torch.exp(0.5 * self.log_vars(features))
        else:
            stds = torch.exp(0.5 * self.log_var)
        normal_dist = normal.Normal(means, stds)
        return torch.distributions.transformed_distribution.TransformedDistribution(normal_dist, [torch.distributions.transforms.TanhTransform(cache_size=1)])

    def sample(self, states):
        dist = self.forward_dist(states)
        return dist.sample()

class IQLTrainer():
    def __init__(self, **kwargs):
        self.device = kwargs["device"]

        self.env_name = kwargs["env_name"]
        self.env = gym.make(self.env_name)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.dataset = self.env.get_dataset()

        self.states = torch.tensor(self.dataset["observations"], device=self.device)
        self.actions = torch.tensor(self.dataset["actions"], device=self.device)
        self.rewards = torch.tensor(self.dataset["rewards"], device=self.device)
        # Done in the official IQL implementation for antmaze
        if "antmaze" in self.env_name:
            self.rewards -= 1
        self.terminals = torch.tensor(self.dataset["terminals"], device=self.device)

        self.n = self.states.size()[0]
        self.probs = torch.ones(self.n - 1, device=self.device)

        kwargs["q_net_args"]["state_dim"] = self.state_dim
        kwargs["q_net_args"]["action_dim"] = self.action_dim
        kwargs["q_net_args"]["type"] = 'Q'
        kwargs["v_net_args"]["state_dim"] = self.state_dim
        kwargs["v_net_args"]["action_dim"] = self.action_dim
        kwargs["v_net_args"]["type"] = 'V'
        kwargs["policy_args"]["state_dim"] = self.state_dim
        kwargs["policy_args"]["action_dim"] = self.action_dim

        self.batch_size = kwargs["batch_size"]
        self.n_critic_updates = kwargs["n_critic_updates"]
        self.n_actor_updates = kwargs["n_actor_updates"]
        self.model_save_interval = kwargs["save_interval"]
        self.model_save_dir = kwargs["save_dir"]
        self.model_save_name = kwargs["save_name"]
        self.verbose = kwargs["verbose"]
        self.q_net_1_save_path = os.path.join(self.model_save_dir, self.model_save_name + "_q_net_1.pt")
        self.q_net_2_save_path = os.path.join(self.model_save_dir, self.model_save_name + "_q_net_2.pt")
        self.v_net_save_path = os.path.join(self.model_save_dir, self.model_save_name + "_v_net.pt")
        self.policy_net_save_path = os.path.join(self.model_save_dir, self.model_save_name + "_policy_net.pt")
        self.visualize = kwargs["visualize"]
        self.vis_points = kwargs["vis_points"]
        self.figures_dir = kwargs["figures_dir"]
        self.figures_interval = kwargs["figures_interval"]
        self.eval_interval = kwargs["eval_interval"]
        self.eval_samples = kwargs["eval_samples"]
        self.log_save_path = "../data/{}.log".format(self.model_save_name)
        self.joint = kwargs["joint"]

        self.q_net_1 = TDNetwork(**kwargs["q_net_args"])
        self.target_q_net_1 = deepcopy(self.q_net_1)
        self.q_net_2 = TDNetwork(**kwargs["q_net_args"])
        self.target_q_net_2 = deepcopy(self.q_net_2)
        self.q_lr = kwargs["q_lr"]

        self.q_net_1.to(self.device)
        self.q_net_2.to(self.device)
        self.q_optimizer_1 = kwargs["q_opt_class"](self.q_net_1.parameters(), lr=self.q_lr)
        self.q_optimizer_2 = kwargs["q_opt_class"](self.q_net_2.parameters(), lr=self.q_lr)
        self.target_q_net_1.to(self.device)
        self.target_q_net_2.to(self.device)

        self.v_net = TDNetwork(**kwargs["v_net_args"])
        self.v_lr = kwargs["v_lr"]
        self.v_net.to(self.device)
        self.v_optimizer = kwargs["v_opt_class"](self.v_net.parameters(), lr=self.v_lr)

        self.policy_net = Policy(**kwargs["policy_args"])
        self.policy_net.to(self.device)
        self.policy_lr = kwargs["policy_lr"]
        self.policy_optimizer = kwargs["policy_opt_class"](self.policy_net.parameters(), lr=self.policy_lr)
        if self.joint:
#            actor_epochs = int(self.n_critic_updates * self.batch_size / self.n)
            actor_epochs = self.n_critic_updates
        else:
            actor_epochs = int(self.n_actor_updates * self.batch_size / self.n)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.policy_optimizer, actor_epochs)

        self.gamma = kwargs["gamma"]
        self.expectile = kwargs["expectile"]
        self.polyak = kwargs["polyak"]
        self.beta = kwargs["beta"]

    def expectile_loss(self, value, expectile_prediction):
        u = value - expectile_prediction
        return torch.mean(torch.abs(self.expectile * torch.square(u) + (nn.functional.relu(-1 * u) * u)))

    def square_loss(self, value, mean_prediction):
        return torch.mean(torch.square(value - mean_prediction))

    def L_V(self, states, actions):
        Q_input = torch.cat([states, actions], dim=-1)
        Q_1 = self.target_q_net_1.forward(Q_input)
        Q_2 = self.target_q_net_2.forward(Q_input)
        Q_values = torch.minimum(Q_1, Q_2)
        V_values = self.v_net.forward(states)
        return self.expectile_loss(Q_values, V_values)

    def L_Q(self, q_net, states, actions, rewards, next_states, terminals):
        Q_values = q_net.forward(torch.cat([states, actions], dim=-1)).squeeze(-1)
        V_values = self.v_net.forward(next_states).squeeze(-1) * torch.logical_not(terminals).to(torch.float)
        return self.square_loss(rewards + self.gamma * V_values, Q_values)

    def polyak_update(self):
        for target_param, param in zip(self.target_q_net_1.parameters(), self.q_net_1.parameters()):
            target_param.data.copy_((1.0 - self.polyak) * target_param.data + self.polyak * param.data)
        for target_param, param in zip(self.target_q_net_2.parameters(), self.q_net_2.parameters()):
            target_param.data.copy_((1.0 - self.polyak) * target_param.data + self.polyak * param.data)

    def _TD_networks_update(self, sample_states, sample_actions, sample_rewards, sample_next_states, sample_terminals):
        L_V = self.L_V(sample_states, sample_actions)
        self.v_optimizer.zero_grad()
        L_V.backward()
        self.v_optimizer.step()

        L_Q_1 = self.L_Q(self.q_net_1, sample_states, sample_actions, sample_rewards, sample_next_states, sample_terminals)
        self.q_optimizer_1.zero_grad()
        L_Q_1.backward()
        self.q_optimizer_1.step()

        L_Q_2 = self.L_Q(self.q_net_2, sample_states, sample_actions, sample_rewards, sample_next_states, sample_terminals)
        self.q_optimizer_2.zero_grad()
        L_Q_2.backward()
        self.q_optimizer_2.step()

        self.polyak_update()

        return L_V, L_Q_1, L_Q_2

    def train_TD_networks(self):
        for update in range(self.n_critic_updates):
            sample_indices = torch.multinomial(self.probs, self.batch_size, replacement=False)
            sample_states = self.states[sample_indices]
            sample_actions = self.actions[sample_indices]
            sample_rewards = self.rewards[sample_indices]
            sample_next_states = self.states[sample_indices + 1]
            sample_terminals = self.terminals[sample_indices]

            L_V, L_Q_1, L_Q_2 = self._TD_networks_update(sample_states, sample_actions, sample_rewards, sample_next_states, sample_terminals)

            if self.verbose:
                print("update: {}, L_V: {}, L_Q_1: {}, L_Q_2: {}".format(update, L_V.item(), L_Q_1.item(), L_Q_2.item()))

            if update % self.figures_interval == 0 and self.visualize:
                self.visualize_maze_values(self.vis_points, update)
                if self.verbose:
                    print("save figure")

            if update % self.model_save_interval == 0 and update != 0:
                torch.save(self.q_net_1.state_dict(), self.q_net_1_save_path)
                torch.save(self.q_net_2.state_dict(), self.q_net_2_save_path)
                torch.save(self.v_net.state_dict(), self.v_net_save_path)
                if self.verbose:
                    print("q nets and v net saved")

    def load_TD_networks(self, q_net_1_path, q_net_2_path, v_net_path):
        self.q_net_1.load_state_dict(torch.load(q_net_1_path))
        self.q_net_2.load_state_dict(torch.load(q_net_2_path))
        self.v_net.load_state_dict(torch.load(v_net_path))

    def L_pi(self, states, actions):
        Q_input = torch.cat([states, actions], dim=-1)
        Q_1 = self.target_q_net_1.forward(Q_input)
        Q_2 = self.target_q_net_2.forward(Q_input)
        Q_values = torch.minimum(Q_1, Q_2)
        V_values = self.v_net.forward(states)
        exp_advantages = torch.clip(torch.exp(self.beta * (Q_values - V_values)), max=100)
        action_log_probs = self.policy_net.forward_dist(states).log_prob(torch.clamp(actions, min=-0.99, max=0.99))
        return -1 * torch.mean(exp_advantages * action_log_probs)

    def _policy_update(self, sample_states, sample_actions):
        L_pi = self.L_pi(sample_states, sample_actions)
        self.policy_optimizer.zero_grad()
        L_pi.backward()
        self.policy_optimizer.step()

        return L_pi

    def train_policy(self):
        for update in range(self.n_actor_updates):
            sample_indices = torch.multinomial(self.probs, self.batch_size, replacement=False)
            sample_states = self.states[sample_indices]
            sample_actions = self.actions[sample_indices]

            L_pi = self._policy_update(sample_states, sample_actions)

            if self.verbose:
                print("update: {}, L_pi: {}".format(update, L_pi.item()))

            if update % self.model_save_interval == 0 and update != 0:
                torch.save(self.policy_net.state_dict(), self.policy_net_save_path)
                if self.verbose:
                    print("policy net saved")

            if update % int(self.n / self.batch_size) == 0 and update != 0:
                self.scheduler.step()

    def train_joint(self):
        for update in range(self.n_critic_updates):
            sample_indices = torch.multinomial(self.probs, self.batch_size, replacement=False)
            sample_states = self.states[sample_indices]
            sample_actions = self.actions[sample_indices]
            sample_rewards = self.rewards[sample_indices]
            sample_next_states = self.states[sample_indices + 1]
            sample_terminals = self.terminals[sample_indices]

            L_V, L_Q_1, L_Q_2 = self._TD_networks_update(sample_states, sample_actions, sample_rewards, sample_next_states, sample_terminals)
            L_pi = self._policy_update(sample_states, sample_actions)

            if self.verbose:
                print("update: {}, L_V: {}, L_Q_1: {}, L_Q_2: {}, L_pi: {}".format(update, L_V.item(), L_Q_1.item(), L_Q_2.item(), L_pi.item()))

            if update % self.figures_interval == 0 and self.visualize:
                self.visualize_maze_values(self.vis_points, update)
                if self.verbose:
                    print("save figure")

            if update % self.eval_interval == 0:
                avg_returns, avg_length, avg_success = self.eval()
                with open(self.log_save_path, 'a') as f:
                    f.write("update: {}, average returns: {}, average length: {}, average success: {}\n".format(update, avg_returns, avg_length, avg_success))

            if update % self.model_save_interval == 0 and update != 0:
                torch.save(self.q_net_1.state_dict(), self.q_net_1_save_path)
                torch.save(self.q_net_2.state_dict(), self.q_net_2_save_path)
                torch.save(self.v_net.state_dict(), self.v_net_save_path)
                torch.save(self.policy_net.state_dict(), self.policy_net_save_path)
                if self.verbose:
                    print("networks saved")

#            if update % int(self.n / self.batch_size) == 0 and update != 0:
            self.scheduler.step()

    def eval(self):
        avg_returns = 0
        avg_length = 0
        avg_success = 0
        with torch.no_grad():
            for _ in range(self.eval_samples):
                state, done = self.env.reset(), False
                returns = 0
                length = 0
                while not done:
                    action = self.policy_net.sample(torch.tensor(state, device=self.device, dtype=torch.float)).cpu().numpy()
                    next_state, reward, done, info = self.env.step(action)
                    returns += reward
                    length += 1

                    state = next_state

                avg_returns += returns / self.eval_samples
                avg_length += length / self.eval_samples
                if returns > 0:
                    avg_success += 1 / self.eval_samples

        return avg_returns, avg_length, avg_success

    def visualize_maze_values(self, n_points, update):
        sample_indices = torch.multinomial(self.probs, n_points, replacement=False)
        sample_states = self.states[sample_indices]
        V_values = np.maximum(np.abs(self.v_net.forward(sample_states).cpu().detach().numpy().flatten()), 0.001)
        np_indices = sample_indices.cpu().detach().numpy()
        np_states = sample_states.cpu().detach().numpy()

#        inliers = (np.abs(V_values - np.median(V_values)) < 4.0 * np.std(V_values)).flatten()
#        np_states = np_states[inliers]
#        V_values = V_values[inliers]
        order = np.argsort(V_values)
        np_states = np_states[order]
        V_values = V_values[order]

        x_values = np_states[:, 0]
        y_values = np_states[:, 1]
        fig = plt.figure()
        norm = colors.LogNorm(vmin=np.min(V_values), vmax=np.max(V_values))
        plt.scatter(x_values, y_values, c=V_values, norm=norm)
        plt.colorbar()
        plt.savefig(os.path.join(self.figures_dir, "{}_{}.png".format(self.model_save_name, update)))
        plt.close(fig)

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
    cuda_num = 0
    iql_test(cuda_num)
