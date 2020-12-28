# Custom DQN, DDQN, Multi-Step DDQN Implementation

This is a simple implementation of a DQN with some of its improvements, tested in CartPole-v0 enviroment, in Python language. The libraries I've used in the project are:

1. Tensorflow
1. Keras
1. Numpy
1. Matplotlib
1. OpenAI-Gym

# Deep Q-Network (DQN)
I have used the sequential model of Keras to constuct a simple. yet effective Q-Network. It consists of:

1. N Hidden Layers of M Hidden Units each Layer.
1. The output of each Hidden Layer passes from a 'Relu' activation funciton.
1. The output layer uses a softmax activation function.
1. Uses 'Adam' optimizer for minimizing the loss.
1. Uses 'Huber' loss for measuring the training error.

Contains a method for predicting Q-Values for given states and a method for fitting experiences into the network.

# Memory Buffer
Our memory buffer is a Uniform Replay Memory buffer. It is used for storing experiences of the game. An experience is stored as:
(s, a, r, 's, t) --> (State, Action, Reward, Next_State, Terminal). The buffer size is limited. It contains a method for storing experiences (samples)
and a method for returning a random batch.

# Agent
1. Uses Decaying Epsilon Greedy Policy.
1. Uses an Online DQN for training and a Target DQN for making predictions. 
1. Uses soft-update strategy for updating the weights of the Target DQN.
1. Uses a Uniform Replay Memory.
1. Uses a training delay (Minimum number of steps required before the training starts).
1. Uses a discount (gamma) factor for the collection of rewards.

The DQN Algorithm runs very fast, but it is not guaranteed that It will converge fast. You might want to try the Double DQN (DDQN) or even the Multi-Step Double DQN.
In the image below, You can see how DQN slowly converges in the CartPole-v0 enviroment for 500 episodes.

![DQN-vis](https://github.com/kochlisGit/Deep-Reinforcement-Learning/blob/master/Custom%20DQN/dqn_average_return.png)
