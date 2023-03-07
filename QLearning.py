import numpy as np
import random
from numpy import savetxt
import sys
import matplotlib.pyplot as plt

#
# This class implements the Q-Learning algorithm.
# We can use this implementation to solve Toy text environments from Gym project. 
#

class QLearning:

    def __init__(self, env, alpha, gamma, epsilon, epsilon_min, epsilon_dec, episodes):
        self.env = env
        self.q_table = np.zeros([env.observation_space.n, env.action_space.n])
        self.alpha = alpha                  # Taxa de aprendizado , quão maior, maior valor se da ao aprendizado.
        self.gamma = gamma                  # O quão relevante são as ecompensas futuras em relação a atual 
        self.epsilon = epsilon              # Chance de escolha de ação aleatória 
        self.epsilon_min = epsilon_min
        self.epsilon_dec = epsilon_dec
        self.episodes = episodes

    def select_action(self, state):
        rv = random.uniform(0, 1)
        if rv < self.epsilon:
            return self.env.action_space.sample()           # Explore action space
        return np.argmax(self.q_table[state])               # Exploit learned values
    
    def select_random_action(self):
        return self.env.action_space.sample() # Explore action space

    def train(self):
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
                #action = self.select_random_action()
                next_state, reward, done, truncated, _ = self.env.step(action)     # Executa uma ação
        
                # Itera sobre Q-table:
                old_value = self.q_table[state,action]                             # Valor da ação escolhida no estado atual 
                next_max = np.max(self.q_table[next_state])                        # Melhor valor de um estado futuro
                
                # Atualiza o valor do estado atual considerando o Algorítimo Q-learning 
                new_value = old_value + self.alpha*(reward + self.gamma*next_max - old_value)             
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

        return self.q_table , reward_per_episode
   
