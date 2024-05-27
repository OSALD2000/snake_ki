from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import numpy as np
import tensorflow as tf
from collections import deque
import random
from Actions import ACTIONS


class Agent():
    def __init__(self, state_size, action_size, feature_size):
        self.state_size = state_size
        self.feature_size = feature_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(self.feature_size,activation='relu', input_shape=(self.feature_size,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse')
        return model
    

    def act(self, env):
        action_id = np.argmax(self.model.predict(env))
        ACTIONS[action_id]()
        return action_id
    
    def update_q_values(self, env, action, reward, new_env, learning_rate, discount_rate):
        
        target = reward + discount_rate * np.max(self.model.predict(new_env))

        current_q_values = self.model.predict(env)

        current_q_values[0][action] = (1 - learning_rate) * current_q_values[0][action] + learning_rate * target

        self.model.fit(env, current_q_values, epochs=1, verbose=0)