from environment import LunarLander
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tensorflow.keras.losses import Huber
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.networks.categorical_q_network import CategoricalQNetwork
from tf_agents.agents.categorical_dqn.categorical_dqn_agent import CategoricalDqnAgent
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.metrics.tf_metrics import AverageReturnMetric
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.policies import random_tf_policy
import matplotlib.pyplot as plt
import time

# 1. Creating tf-environments (Train: for training, Eval: For testing)
train_env = TFPyEnvironment( environment=LunarLander() )
eval_env = TFPyEnvironment( environment=LunarLander() )

# 2. Constructing the Categorical QNetworks: Online & Target.
# Default Activation Function: "relu".
# Default Weight Initialization: "He (Xavier) Initialization".
hidden_units = [256, 256]
num_atoms = 51

online_q_net = CategoricalQNetwork(
    input_tensor_spec=train_env.observation_spec(),
    action_spec=train_env.action_spec(),
    conv_layer_params=None,
    fc_layer_params=hidden_units,
    num_atoms=num_atoms
)
target_q_net = CategoricalQNetwork(
    input_tensor_spec=train_env.observation_spec(),
    action_spec=train_env.action_spec(),
    conv_layer_params=None,
    fc_layer_params=hidden_units,
    num_atoms=num_atoms
)

# Defining train_step, which will be used to store the current step.
train_step = tf.Variable(initial_value=0)
total_steps = 100000

# Defining decay epsilon-greedy strategy.
decay_epsilon_greedy = PolynomialDecay(
    initial_learning_rate=1.0,
    decay_steps=total_steps,
    end_learning_rate=0.001,
)

# 3. Constructing the DQN Agent.
adam_optimizer = Adam(learning_rate=0.00025)
n_steps = 3
tau = 0.001
huber_loss = Huber()
g = 0.99
min_q = -20
max_q = 20

agent = CategoricalDqnAgent(
    time_step_spec=train_env.time_step_spec(),
    action_spec=train_env.action_spec(),
    categorical_q_network=online_q_net,
    optimizer=adam_optimizer,
    min_q_value=min_q,
    max_q_value=max_q,
    epsilon_greedy=lambda: decay_epsilon_greedy(train_step),
    n_step_update=n_steps,
    target_categorical_q_network=target_q_net,
    target_update_tau=tau,
    target_update_period=1,
    td_errors_loss_fn=huber_loss,
    gamma=g,
    train_step_counter=train_step
)
agent.initialize()

# 4. Constructing the Replay Memory.
memory_size = 50000

replay_buffer = TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=memory_size
)

# Initializing Observer of replay buffer to store experiences (trajectories) to memory.
replay_buffer_observer = replay_buffer.add_batch

# Defining Metrics for measuring training progress.
train_metrics = [ AverageReturnMetric() ]

# 5. Defining intial policy as random to collect enough examples to fill the memory buffer (Training delay).
initial_collect_policy = random_tf_policy.RandomTFPolicy( train_env.time_step_spec(), train_env.action_spec() )
training_delay = 1000

class ShowProgress:
    def __init__(self, total):
        self.counter = 0
        self.total = total
    def __call__(self, trajectory):
        if not trajectory.is_boundary():
            self.counter += 1
        if self.counter % 100 == 0:
            print("\r{}/{}".format(self.counter, self.total), end="")

init_driver = DynamicStepDriver(
    env=train_env,
    policy=initial_collect_policy,
    observers=[ replay_buffer.add_batch, ShowProgress(training_delay) ],
    num_steps=training_delay
)

# Collecting experiences.
init_driver.run()

# 6. Training the agent.
dataset = replay_buffer.as_dataset(sample_batch_size=64, num_steps=n_steps+1, num_parallel_calls=3).prefetch(3)

all_train_loss = []
all_metrics = []

collect_driver = DynamicStepDriver(
    env=train_env,
    policy=agent.collect_policy,
    observers=[replay_buffer_observer] + train_metrics,
    num_steps=n_steps+1
)

def train_agent(num_steps):
    time_step = None
    policy_state = agent.collect_policy.get_initial_state(train_env.batch_size)
    dataset_iter = iter(dataset)

    for step in range(num_steps):
        current_metrics = []

        time_step, policy_state = collect_driver.run(time_step, policy_state)
        trajectories, buffer_info = next(dataset_iter)

        train_loss = agent.train(trajectories)
        all_train_loss.append( train_loss.loss.numpy() )

        for i in range( len(train_metrics) ):
            current_metrics.append(train_metrics[i].result().numpy())

        all_metrics.append(current_metrics)

        if step % 500 == 0:
            print( "\nIteration: {}, loss:{:.2f}".format( step, train_loss.loss.numpy() ) )

            for i in range( len(train_metrics) ):
                print( '{}: {}'.format( train_metrics[i].name, train_metrics[i].result().numpy() ) )

start = time.time()
train_agent(total_steps)
end = time.time()

# 7. Plotting metrics and results.
average_return = [ metric[0] for metric in all_metrics ]
plt.plot(average_return)
plt.show()

# 8. Evaluating results.
def evaluate(env, policy, num_episodes=10):
    total_return = 0.0

    for _ in range(num_episodes):
        time_step = env.reset()
        episode_return = 0.0
        step = 0
        print("Step: 0")
        while not time_step.is_last():
            step += 1
            print("---\nStep: {}".format(step))
            action_step = policy.action(time_step)
            print("Action taken: {}".format(action_step.action))
            time_step = env.step(action_step.action)
            episode_return += time_step.reward
            print("Reward: {} \n".format(episode_return))

        total_return += episode_return

    avg_return = total_return / num_episodes

    return avg_return.numpy()[0]

# Reset the train step
agent.train_step_counter.assign(0)

#reset eval environment
eval_env.reset()

# Evaluate the agent's policy once before training.
avg_return = evaluate(eval_env, agent.policy, 10)

print('\nAverage return in 10 episodes =', avg_return)
print('Total execution time in seconds =', end-start)