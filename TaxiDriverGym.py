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
# python TaxiDriverGym.py algoritimo reuse_data

# - algoritimo = q (qlearning) , sarsa ou both (comparação entre sarsa e qlearning)
# - reuse_data =  1(Usar csv já existente) , 0(Treinar algorítimo novamente) 

# ------------------------------------------------


def run_algorithm(env, q_table):
    
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


def main(previous_info , algo):
    
    # Ambiente:
    env = gym.make("Taxi-v3", render_mode='ansi').env
    
    # Nome csv e do Grafico
    csv_name = 'data/'+ algo + '-taxi-driver.csv'
    grafic_actions_name = 'results/'+ algo + 'TaxiDriver_actions_per_episode'
    
    # Variávies do modelo:
    alpha = 0.1
    gamma = 0.99
    epsilon = 0.7
    epsilon_min = 0.05
    epsilon_dec = 0.99
    episodes = 5000
    
    if(previous_info == "1"):
         # Descomentar para utilizar q_table já treinada

        if (algo in ['q' , 'sarsa']):
            q_table = loadtxt(csv_name, delimiter=',')
            
            # Roda algoritimo para verificar aprendizagem:
            run_algorithm(env, q_table)
        else:
             raise Exception("[ERROR] Wrong inputs to the program.") 

    else:
        # Executa algorítimo de aprendizagem escolhido.  
        
        if(algo == 'q'):
            qlearning = QLearning(env, alpha=alpha, gamma=gamma, epsilon=epsilon, epsilon_min=epsilon_min, epsilon_dec=epsilon_dec, episodes=episodes)
            q_table , rewards_list =  qlearning.train(csv_name, grafic_actions_name)
            
            # Roda algoritimo para verificar aprendizagem:
            print('\n-------------------------------------')
            print('------- RUNNING Q-LEARNING ----------')
            print('-------------------------------------\n')
            run_algorithm(env, q_table)
  
        elif(algo == 'sarsa'):

            sarsa = Sarsa(env, alpha=alpha, gamma=gamma, epsilon=epsilon, epsilon_min=epsilon_min, epsilon_dec=epsilon_dec, episodes=episodes)
            q_table , rewards_list = sarsa.train(csv_name, grafic_actions_name)
            
            # Roda algoritimo para verificar aprendizagem:
            print('\n-------------------------------------')
            print('---------- RUNNING SARSA ------------')
            print('-------------------------------------\n')
            run_algorithm(env, q_table)

        else:

            if(algo == 'both'):
                
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

                run_algorithm(env, q_table_q )

                print('\n-------------------------------------')
                print('---------- RUNNING SARSA ------------')
                print('-------------------------------------\n')

                run_algorithm(env, q_table_sarsa)

                # Gera grafico de comparação dos algorítimos:
                plot_compare_algorithms(q_rewards_list, sarsa_rewards_list, episodes)

            else:
                raise Exception(" [ERROR] Wrong inputs to the program.") 
                 
if __name__ == '__main__':
    
    algoritimo =   sys.argv[1]
    reuse_data =   sys.argv[2]
    
    main(reuse_data, algoritimo)