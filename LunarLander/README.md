# LunarLander with tf-agents

This is an implementation of a multi-step DQN agent to solve the Lunar Lander problem.

https://gym.openai.com/envs/LunarLander-v2/

The agent is writen in python. Also, I used OpenAI's gym to create the environment of the game, because I don't like the suite_gym of the tf-agent's library, so I had to fit it into the tf's enviroment.

# Agent's Properties

1. 3-Step DQN Agent.
1. DQN with 2 hidden layers of 256 units each. 'relu' activation function at the output of each layer. 'He Initialization method of the weights'.
1. Both Online & Target network used with tau = 0.001 and update_period = 1 step.
1. gamma = 0.99
1. Decaying epsilon greedy strategy starting from 1.0 and ending to 0.001 after 100000 training steps.
1.  Uniform Replay Memory

To run the code, You will need the following libraries:

1. tensorflow
1. tf-agents
1. gym
1. numpy
1. matplotlib

The performance of the agent is shown in the graph below, where the blue line represents the average return per steps.

![Plot](https://github.com/kochlisGit/Deep-Reinforcement-Learning/blob/master/LunarLander/dqn_average_return.png)
