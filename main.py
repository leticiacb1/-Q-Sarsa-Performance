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
from QLearning import *
from Sarsa import *


# ----------------- DOCUMENTAÇÃO  -----------------
# Como rodar:
# python main.py ambiente algoritimo reuse_data

# Valores possíveis:
# - ambiente = taxi or cliff
# - algoritimo = q (qlearning) , sarsa ou both (comparação entre sarsa e qlearning)
# - reuse_data =  1(Usar csv já existente) , 0(Treinar algorítimo novamente) 

# ------------------------------------------------


def run_algorithm(board, q_table):

    env = gym.make(board, render_mode='human').env
    
    (state, _) = env.reset()
    rewards , epochs , actions = 0 , 0 , 0
    done = False
    frames = [] # for animation
        
    while (not done) and (epochs < 100):
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
        epochs += 1
        
    if (epochs == 100):
        raise Exception("[ERROR] I didn't find the optimal solution !") 

    
    clear_output(wait=True)
    print_frames(frames)

    print('\n-------------------------------------')
    print('           SHOW METRICS              ')
    print('-------------------------------------\n')

    print("\n")
    print("   > Actions taken: {}".format(actions))
    print("\n   > Rewards: {}".format(rewards))


def main(ambiente, previous_info , algoritimo):
    
    # Variávies do modelo:
    alpha = 0.1
    gamma = 0.99
    epsilon = 0.7
    epsilon_min = 0.05
    epsilon_dec = 0.99
    episodes = 5000

    # Ambiente:
    if(ambiente == 'taxi'):
        board = "Taxi-v3"
        env = gym.make(board, render_mode='ansi').env

        # Nome csv e do Grafico
        csv_name = 'data/'+ algoritimo + '-taxi-driver.csv'
        grafic_actions_name = 'results/'+ algoritimo + 'TaxiDriver_actions_per_episode'

    elif(ambiente == 'cliff'):
        board = "CliffWalking-v0"
        env = gym.make(board, render_mode='ansi').env
        
        # Nome csv e do Grafico
        csv_name = 'data/'+ algoritimo + '-cliff-walking.csv'
        grafic_actions_name = 'results/'+ algoritimo + 'CliffWalking_actions_per_episode'
    else:
        raise Exception("[ERROR] Wrong env select.") 
    
    if(previous_info == "1"):
         # Utilizar csv existente

        if (algoritimo in ['q' , 'sarsa']):
            q_table = loadtxt(csv_name, delimiter=',')
            
            # Roda algoritimo para verificar aprendizagem:
            run_algorithm(env, q_table)
        else:
            raise Exception("[ERROR] Wrong inputs to the program.") 

    else:
        # Executa algorítimo de aprendizagem escolhido.  
        
        if(algoritimo == 'q'):
            qlearning = QLearning(env, alpha=alpha, gamma=gamma, epsilon=epsilon, epsilon_min=epsilon_min, epsilon_dec=epsilon_dec, episodes=episodes)
            q_table , rewards_list =  qlearning.train(csv_name, grafic_actions_name)
            
            # Roda algoritimo para verificar aprendizagem:
            print('\n-------------------------------------')
            print('------- RUNNING Q-LEARNING ----------')
            print('-------------------------------------\n')

            run_algorithm(board, q_table)
  
        elif(algoritimo == 'sarsa'):

            sarsa = Sarsa(env, alpha=alpha, gamma=gamma, epsilon=epsilon, epsilon_min=epsilon_min, epsilon_dec=epsilon_dec, episodes=episodes)
            q_table , rewards_list = sarsa.train(csv_name, grafic_actions_name)
            
            # Roda algoritimo para verificar aprendizagem:
            print('\n-------------------------------------')
            print('---------- RUNNING SARSA ------------')
            print('-------------------------------------\n')
            run_algorithm(board, q_table)

        else:

            if(algoritimo == 'both'):
                
                # Algorítimos:
                q =   QLearning(env, alpha=alpha, gamma=gamma, epsilon=epsilon, epsilon_min=epsilon_min, epsilon_dec=epsilon_dec, episodes=episodes)
                sarsa =  Sarsa(env, alpha=alpha, gamma=gamma, epsilon=epsilon, epsilon_min=epsilon_min, epsilon_dec=epsilon_dec, episodes=episodes) 
                
                # Treino:
                q_table_q , q_rewards_list =  q.train('', '', both = True)
                q_table_sarsa , sarsa_rewards_list =  sarsa.train('', '' , both = True)

                # Roda algoritimo para verificar aprendizagem:
                print('\n-------------------------------------')
                print('------- RUNNING Q-LEARNING ----------')
                print('-------------------------------------\n')

                run_algorithm(board, q_table_q)

                print('\n-------------------------------------')
                print('---------- RUNNING SARSA ------------')
                print('-------------------------------------\n')

                run_algorithm(board, q_table_sarsa)

                # Gera grafico de comparação dos algorítimos:
                plot_compare_algorithms(q_rewards_list, sarsa_rewards_list, episodes, ambiente)

            else:
                raise Exception(" [ERROR] Wrong inputs to the program.") 
                 
if __name__ == '__main__':
    
    ambiente = sys.argv[1]
    algoritimo =   sys.argv[2]
    reuse_data =   sys.argv[3]
    
    main(ambiente, reuse_data, algoritimo)