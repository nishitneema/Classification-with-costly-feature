import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape
from tensorflow.keras.optimizers import Adam
import pandas as pd
from config import config

from rl.agents import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

from env_mnist import SeqEnvironment


def build_model(input_shape, output_shape, NN_SIZE = 512):
    model = Sequential()
    model.add(tf.keras.layers.InputLayer((1,1000,2,784)))
    model.add(Reshape(target_shape=(-1,input_shape)))
    model.add(Dense(NN_SIZE, activation='relu'))
    model.add(Dense(NN_SIZE, activation='relu'))
    model.add(Dense(NN_SIZE, activation='relu'))
    model.add(Dense(output_shape, activation='linear'))
    model.add(Flatten())
    return model


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')
x_train = x_train.reshape(x_train.shape[0],-1) 

data_train = np.concatenate((x_train, y_train.reshape(-1,1)), axis=1)
print(data_train.shape)

costs = -1*np.ones(x_train.shape[1])

env = SeqEnvironment()



model = build_model(2*784, 794)
model.build(input_shape=(1,1000,2,784))

policy = EpsGreedyQPolicy()
memory = SequentialMemory(limit=50000, window_length=1)
dqn = DQNAgent(model=model, memory=memory, policy=policy, nb_actions=config.ACTION_DIM, nb_steps_warmup=10, target_model_update=1e-2)

dqn.compile(tf.keras.optimizers.legacy.Adam(learning_rate=1e-3), metrics=['mae'])

dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)