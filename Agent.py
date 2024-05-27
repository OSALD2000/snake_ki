from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import numpy as np
import tensorflow as tf
from collections import deque
import random
import subprocess
from enum import Enum
from functools import partial

class KEYS(Enum):
    LEFT = "Left"
    RIGHT = "Right"
    UP = "Up"
    DOWN = "Down"

def send_key_press(key):
    subprocess.call(["xdotool", "key", key])

ACTIONS = {
    0:    partial(send_key_press, KEYS.LEFT.value),
    1:   partial(send_key_press, KEYS.RIGHT.value),
    2:      partial(send_key_press, KEYS.UP.value),
    3:    partial(send_key_press, KEYS.DOWN.value),
}

class Agent():
    def __init__(self, state_size, action_size, feature_size):
        self.state_size = state_size
        self.feature_size = feature_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(self.feature_size)),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate), loss='mse')
        return model
    

    def act(self, state):

        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        return np.argmax(self.model.predict(state))
    
    def train(self):
        
        pass 
    