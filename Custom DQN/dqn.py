from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import Input
from tensorflow.keras.optimizers import Adam

# Deep-Q Network Implementation.
class DQN:
    def __init__(self, input_size, hidden_units, output_size, learn_rate):
        self._build(input_size, hidden_units, output_size)
        self._compile(learn_rate)

    # Builds DQN layers.
    # Activation Functions:     'Relu' in hidden layers, 'Softmax' in Output Layer.
    # Initialization Technique: 'Normal Xavier'
    def _build(self, input_size, hidden_units, output_size):
        self.model = Sequential()
        self.model.add( Input(input_size,) )
        for num_of_units in hidden_units:
            self.model.add( Dense(units=num_of_units, activation='relu', kernel_initializer='glorot_normal') )
        self.model.add( Dense(units=output_size, activation='softmax', kernel_initializer='glorot_normal') )

    # Compiles DQN.
    # Optimizer:    'Adam'
    # Loss:         'Huber Loss'
    def _compile(self, learn_rate):
        self.model.compile(
            optimizer=Adam(learning_rate=learn_rate),
            loss='huber_loss',
            metrics=['accuracy']
        )

    # Predicts the Q Values of a state.
    def predict(self, states):
        return self.model.predict(states)

    # Fits the agent's experience into the network.
    def partial_fit(self, states, targets, weights=None):
        self.model.fit(x=states, y=targets, sample_weight=weights, verbose=0)