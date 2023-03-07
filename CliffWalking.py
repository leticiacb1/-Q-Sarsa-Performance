import random
from IPython.display import clear_output
import gymnasium as gym
import numpy as np
from numpy import loadtxt
import matplotlib.pyplot as plt
from IPython.display import clear_output
from time import sleep
import sys

from utils import *
from Sarsa import Sarsa

def main(previous_info):
    env = gym.make("CliffWalking-v0", render_mode='ansi').env
    
    if(previous_info):
        # Descomentar para utilizar q_table já treinada
        q_table = loadtxt('data/sarsa-cliff-walking.csv', delimiter=',')
    else:
         # Executa algorítimo de aprendizagem.   
        qlearn = Sarsa(env, alpha=0.1, gamma=0.99, epsilon=0.7, epsilon_min=0.05, epsilon_dec=0.99, episodes=5000)
        q_table , rewards_list = qlearn.train('results/qsarsa_Actions_per_Episodes')


    (state, _) = env.reset()
    rewards , epochs , actions = 0 , 0 , 0
    done = False
    
    frames = []  # Para animação

    while not done and (epochs < 100):
        print(state)
        action = np.argmax(q_table[state])
        state, reward, done, truncated, info = env.step(action)

        rewards = rewards + reward
        actions = actions + 1
        
        # Put each rendered frame into dict for animation
        frames.append({
            'frame': env.render(),
            'state': state,
            'action': action,
            'reward': reward
            }
        )
        epochs+=1

    clear_output(wait=True)    
    print_frames(frames)


    print("\n")
    print("Actions taken: {}".format(actions))
    print("Rewards: {}".format(rewards))


if __name__ == '__main__':
    
    #algoritimo = sys.argv[1]
    #ambiente = sys.argv[2]

    main(False)
