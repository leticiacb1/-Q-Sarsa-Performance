import numpy as np
import random
from numpy import savetxt
import sys
import matplotlib.pyplot as plt

from Algoritimo import Algoritimo

class Sarsa(Algoritimo):

    def __init__(self, env, alpha, gamma, epsilon, epsilon_min, epsilon_dec, episodes):
        super().__init__(env, alpha, gamma, epsilon, epsilon_min, epsilon_dec, episodes)
    
    def train(self, filename):
        actions_per_episode = []
        reward_per_episode = []
        reward_list = []

        for i in range(1, self.episodes+1):
            (state, _) = self.env.reset()     # Ambiente escolhido aleatoriamente dentre todos os possíveis estados (Número de 0 a 500)
            
            total_rewards = 0 
            reward = 0
            done = False
            actions = 0

            while not done:
                action = self.select_action(state)                                 # Escolhe uma ação
                next_state, reward, done, truncated, _ = self.env.step(action)     # Executa uma ação
        
                # Itera sobre Q-table:
                old_value = self.q_table[state,action]                             # Valor da ação escolhida no estado atual 
                next_q_value = self.q_table[next_state, action]                    # Melhor valor de um estado futuro
                
                # Atualiza o valor do estado atual considerando o Algoritimo Sarsa (on policy) 
                new_value = old_value + self.alpha*(reward + self.gamma*next_q_value - old_value)             
                self.q_table[state, action] = new_value
                
                # Atualiza para o novo estado
                state = next_state
                actions=actions+1
                total_rewards+=reward
            
            # Tratar ruidos do gráfico:
            if(i%10 == 0):
                # Média do total_rewards a cada 10 episodios:
                media = np.mean(reward_list)
                reward_per_episode.append(media)
                reward_list = []
            else:
                reward_list.append(total_rewards)

            #reward_per_episode.append(total_rewards)    
            actions_per_episode.append(actions)
            if i % 100 == 0:
                sys.stdout.write("Episodes: " + str(i) +'\r')
                sys.stdout.flush()
                pass
            if self.epsilon > self.epsilon_min:
                self.epsilon = self.epsilon * self.epsilon_dec
            
            savetxt('data/sarsa-taxi-driver.csv', self.q_table, delimiter=',')
            if (filename is not None): self.plotactions(filename, actions_per_episode, range(0,self.episodes) , 'Actions vs Episodes', 'Episodes', 'Actions')



        return self.q_table , reward_per_episode 
