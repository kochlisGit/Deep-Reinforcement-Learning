# LunarLander with tf-agents

This is an implementation of both a multi-step DQN & a Catigorical c51 agents to solve the Lunar Lander problem.

https://gym.openai.com/envs/LunarLander-v2/

The agent is writen in python. Also, I used OpenAI's gym to create the environment of the game, because I don't like the suite_gym of the tf-agent's library, so I had to fit it into the tf's enviroment.

# DQN Agent's Properties

1. 3-Step DQN Agent.
1. DQN with 2 hidden layers of 256 units each. 'relu' activation function at the output of each layer. 'He Initialization method of the weights'.
1. Both Online & Target network used with tau = 0.001 and update_period = 1 step.
1. gamma = 0.99
1. Decaying epsilon greedy strategy starting from 1.0 and ending to 0.001 after 100000 training steps.
1. Uniform Replay Memory
1. Training Delay (Initial Experience Colletion) = 1000 Steps

To run the code, You will need the following libraries:

1. tensorflow
1. tf-agents
1. gym
1. numpy
1. matplotlib

The performance of the dqn agent is shown in the graph below, where the blue line represents the average return per steps.

![dqn_Plot](https://github.com/kochlisGit/Deep-Reinforcement-Learning/blob/master/LunarLander/dqn_average_return.png)

In comparison with the performance of Catigorical C51 DQN agent.

![c51_Plot](https://github.com/kochlisGit/Deep-Reinforcement-Learning/blob/master/LunarLander/c51_average_return.png)

As you can see, the performance of the 2 agents is not much different. Also, the C51 agent takes slightly more time to train, due to difficult computations. However, in more complex games, such as Pong, C51 Converges a lot faster than DQN, making a asgnificant difference between those 2 agents.

# Conclusion

There is no perfect algorithm that suits best for every problem. However, for simplier tasks, DQN seems to be performing faster and just as the same with better agents.
