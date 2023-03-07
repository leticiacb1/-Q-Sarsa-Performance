import random
from IPython.display import clear_output
import gymnasium as gym
import numpy as np
from numpy import loadtxt
import matplotlib.pyplot as plt
from IPython.display import clear_output
from time import sleep

from utils import *
from QLearning import *

def main(previous_info, ambiente):
    
    if(ambiente == 'taxi'):
        env = gym.make("Taxi-v3", render_mode='ansi').env
    elif(ambiente == 'cliff'):
        env = gym.make("CliffWalking-v0").env


    if(previous_info):
        # Descomentar para utilizar q_table já treinada
        q_table = loadtxt('data/q-learning-taxi-driver.csv', delimiter=',')
    else:
         # Executa algorítimo de aprendizagem.   
        qlearn = QLearning(env, alpha=0.1, gamma=0.99, epsilon=0.7, epsilon_min=0.05, epsilon_dec=0.99, episodes=5000)
        q_table , rewards_list = qlearn.train('results/qlearning_Actions_per_Episodes')


    (state, _) = env.reset()
    epochs, penalties, reward = 0, 0, 0
    done = False
    frames = [] # for animation
        
    while (not done) and (epochs < 100):
        action = np.argmax(q_table[state])
        state, reward, done, t, info = env.step(action)

        if reward == -10:
            penalties += 1

        # Put each rendered frame into dict for animation
        frames.append({
            'frame': env.render(),
            'state': state,
            'action': action,
            'reward': reward
            }
        )
        epochs += 1

    clear_output(wait=True)
        
    print_frames(frames)
    print("\n")
    print("Timesteps taken: {}".format(epochs))
    print("Penalties incurred: {}".format(penalties))

if __name__ == '__main__':
    main(False, 'taxi')

