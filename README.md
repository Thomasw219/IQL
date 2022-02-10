# Pytorch Implementation of Implicit Q-Learning

The paper for Implicit Q-Learning https://arxiv.org/abs/2110.06169

Citation information:

```
@article{kostrikov2021iql,
    title={Offline Reinforcement Learning with Implicit Q-Learning},
    author={Ilya Kostrikov and Ashvin Nair and Sergey Levine},
    year={2021},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

I reimplemented this offline RL algorithm because I'll be using a variation of it in my own research where I am using Pytorch for other components. There exists a Pytorch implemenation in the rlkit repo but I couldn't get docker to work with that repo and rlkit has very old versions of pytorch and other packages.

I'm unsure if the lambda-stack docker image works on non-lambda machines so anyone using might have to replace that with whatever cuda docker image you normally use.
