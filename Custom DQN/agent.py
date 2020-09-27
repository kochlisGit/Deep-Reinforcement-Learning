from dqn import DQN
from memory import UniformReplayMemory
import numpy as np
import random as rand

# Uses Epsilon-Greedy strategy.
# Uses a Uniform Replay Memory.
# Uses 2 DQNs: Online network for predictions & Target network for evaluations.
# Uses Soft Update strategy to update target network's weights.
class DQNAgent:
    def __init__(self, observation_size, num_of_actions, lr=0.00025, tau=0.001,
                 epsilon_max=1.0, epsilon_min=0.01, epsilon_decay=0.0005, gamma=0.85,
                 memory_size=50000, batch_size=64, training_delay=1000):
        self.observation_size = observation_size
        self.online_net = DQN(input_size=observation_size,
                              hidden_units=[20, 20],
                              output_size=num_of_actions,
                              learn_rate=lr)
        self.target_net = DQN(input_size=observation_size,
                              hidden_units=[20, 20],
                              output_size=num_of_actions,
                              learn_rate=lr)
        self.tau = tau

        self.replay_memory = UniformReplayMemory(memory_size)
        self.batch_size = batch_size
        self.training_delay = training_delay

        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon = epsilon_max
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.state = None

    # Manually sets epsilon's value.
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    # Epsilon decays exponentially, based on the current step.
    def decay_epsilon(self, step):
        self.epsilon = self.epsilon_min + \
                       (self.epsilon_max - self.epsilon_min) * np.exp(-self.epsilon_decay * step)

    # Updates the agent's current state in the enviroment.
    def update_state(self, next_state):
        self.state = next_state

    # Adds an experience into the memory.
    def update_memory(self, state, action, reward, next_state, is_terminal):
        sample = (state, action, reward, next_state, is_terminal)
        self.replay_memory.add_sample(sample)

    # Updates Target network's weights.
    def soft_update_target_network(self):
        online_weights = self.online_net.model.get_weights()
        target_weights = self.target_net.model.get_weights()
        target_weights = [ t*(1-self.tau) + o*self.tau for t, o in zip(target_weights, online_weights) ]

    # Exlores a random step.
    def explore(self, env):
        return env.action_space.sample()

    # Exploits the current state.
    def exploit(self):
        Q = self.online_net.predict(self.state)
        return np.argmax(Q, axis=1)[0]

    # Takes an action (Explore / Exploit).
    # The agent uses the epsilon value to determine whether to explore or exploit.
    def take_action(self, env):
        action = None
        p = rand.uniform(0.0, 1.0)
        if p < self.epsilon:
            action = self.explore(env)
        else:
            action = self.exploit()
        next_state, reward, is_terminal, _ = env.step(action)
        next_state = np.reshape( next_state, (1, self.observation_size) )
        return self.state, action, reward, next_state, is_terminal

    # The agent is trained after the minimum required samples are collected.
    # Targets = R + (1-terminal) * g*Q'(s',a'), where a' = argmax( Q'(s') )
    def train_networks(self):
        mini_batch = self.replay_memory.random_batch(self.batch_size)
        states = np.reshape( [sample[0] for sample in mini_batch], (self.batch_size, self.observation_size) )
        actions = [sample[1] for sample in mini_batch]
        rewards = np.array( [sample[2] for sample in mini_batch] )
        next_states = np.reshape( [sample[3] for sample in mini_batch], (self.batch_size, self.observation_size) )
        terminals = np.array( [sample[4] for sample in mini_batch] )

        targets = self.online_net.predict(states)
        next_q = self.target_net.predict(next_states)
        next_actions = np.argmax(next_q, axis=1)

        r = range(self.batch_size)
        updates = rewards + (1 - terminals)*self.gamma*next_q[r, next_actions]
        targets[r, actions] = updates

        self.online_net.partial_fit(states, targets)
        self.soft_update_target_network()

    # Trains the agent. Returns the rewards of each episode.
    def train(self, env, num_of_steps, verbose=1):
        episode_rewards = []
        current_episode_rewards = 0
        self.update_state( np.reshape( env.reset(), (1, self.observation_size) ) )

        for step in range(num_of_steps):
            # The agent takes an action.
            state, action, reward, next_state, is_terminal = self.take_action(env)
            self.update_memory(state, action, reward, next_state, is_terminal)
            self.update_state(next_state)
            current_episode_rewards += reward

            # The agent trains its networks.
            if step > self.training_delay:
                self.train_networks()
                self.decay_epsilon(step-self.training_delay)

            # When an episode ends, reset the environment.
            if is_terminal:
                self.update_state( np.reshape( env.reset(), (1, self.observation_size) ) )
                episode_rewards.append(current_episode_rewards)
                if verbose:
                    print('Step =', step, 'Epsilon =', self.epsilon, ', Reward =', current_episode_rewards)
                current_episode_rewards = 0
        return episode_rewards

class DDQNAgent(DQNAgent):
    def __init__(self, observation_size, num_of_actions, lr=0.00025, tau=0.001,
                 epsilon_max=1.0, epsilon_min=0.01, epsilon_decay=0.0005, gamma=0.85,
                 memory_size=50000, batch_size=64, training_delay=1000):
        super().__init__(observation_size, num_of_actions, lr, tau,
                         epsilon_max, epsilon_min, epsilon_decay, gamma,
                         memory_size, batch_size, training_delay)

    # --- UPDATE ---
    # The agent is trained after the minimum required samples are collected.
    # Targets = R + (1-terminal) * g*Q'(s',a'), where a' = argmax( Q(s') )
    def train_networks(self):
        mini_batch = self.replay_memory.random_batch(self.batch_size)
        states = np.reshape( [sample[0] for sample in mini_batch], (self.batch_size, self.observation_size) )
        actions = [sample[1] for sample in mini_batch]
        rewards = np.array( [sample[2] for sample in mini_batch] )
        next_states = np.reshape( [sample[3] for sample in mini_batch], (self.batch_size, self.observation_size) )
        terminals = np.array( [sample[4] for sample in mini_batch] )

        targets = self.online_net.predict(states)
        next_q = self.target_net.predict(next_states)
        next_actions = np.argmax(self.online_net.predict(next_states), axis=1)

        r = range(self.batch_size)
        updates = rewards + (1 - terminals)*self.gamma*next_q[r, next_actions]
        targets[r, actions] = updates

        self.online_net.partial_fit(states, targets)
        self.soft_update_target_network()

class MSDDQNAgent(DDQNAgent):
    def __init__(self, observation_size, num_of_actions, lr=0.00025, tau=0.001,
                 epsilon_max=1.0, epsilon_min=0.01, epsilon_decay=0.0005, gamma=0.85, n_steps=3,
                 memory_size=50000, batch_size=64, training_delay=1000):
        super().__init__(observation_size, num_of_actions, lr, tau,
                         epsilon_max, epsilon_min, epsilon_decay, gamma,
                         memory_size, batch_size, training_delay)
        self.n_steps = n_steps

    # --- UPDATE ---
    # The agent is trained after the minimum required samples are collected.
    # Targets = R + (1-terminal) * g^(n_steps) *Q'(s',a'), where a' = argmax( Q(s') )
    def train_networks(self):
        mini_batch = self.replay_memory.random_batch(self.batch_size)
        states = np.reshape( [sample[0] for sample in mini_batch], (self.batch_size, self.observation_size) )
        actions = [sample[1] for sample in mini_batch]
        rewards = np.array( [sample[2] for sample in mini_batch] )
        next_states = np.reshape( [sample[3] for sample in mini_batch], (self.batch_size, self.observation_size) )
        terminals = np.array( [sample[4] for sample in mini_batch] )

        targets = self.online_net.predict(states)
        next_q = self.target_net.predict(next_states)
        next_actions = np.argmax(self.online_net.predict(next_states), axis=1)

        r = range(self.batch_size)
        updates = rewards + (1 - terminals) * (self.gamma**self.n_steps) * next_q[r, next_actions]
        targets[r, actions] = updates

        self.online_net.partial_fit(states, targets)
        self.soft_update_target_network()

    # --- UPDATE ---
    # Trains the agent. Returns the rewards of each episode.
    # The Agent performs N steps each time.
    def train(self, env, num_of_steps, verbose=1):
        episode_rewards = []
        current_episode_rewards = 0
        self.update_state( np.reshape( env.reset(), (1, self.observation_size) ) )

        for step in range(num_of_steps):
            state, action, reward, next_state, is_terminal = self.take_action(env)

            # Perform n steps.
            total_reward = reward
            for i in range(1, self.n_steps):
                if is_terminal:
                    break
                self.update_state(next_state)
                _, _, reward, next_state, is_terminal = self.take_action(env)
                total_reward += (self.gamma**i) * reward
            self.update_memory(state, action, total_reward, next_state, is_terminal)
            self.update_state(next_state)
            current_episode_rewards += reward

            # The agent trains its networks.
            if step > self.training_delay:
                self.train_networks()
                self.decay_epsilon(step-self.training_delay)

            # When an episode ends, reset the environment.
            if is_terminal:
                self.update_state( np.reshape( env.reset(), (1, self.observation_size) ) )
                episode_rewards.append(current_episode_rewards)
                if verbose:
                    print('Step =', step, 'Epsilon =', self.epsilon, ', Reward =', current_episode_rewards)
                current_episode_rewards = 0
        return episode_rewards
