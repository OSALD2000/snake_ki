from keras.models import Sequential
from keras.layers import Dense
import numpy as np

class Agent():
    def __init__(self,state, action, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.state = state
        self.action = action
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer='adam')
        return model
    
    def act(self, state):
        return