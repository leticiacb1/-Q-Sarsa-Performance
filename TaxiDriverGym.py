import random
from IPython.display import clear_output
import gymnasium as gym
import numpy as np
from QLearning import QLearning
from numpy import loadtxt
import matplotlib.pyplot as plt

def plotactions(plotFile, data, x_label, y_label , titulo):
    episodes_list = range(0, 5000, 10) 
    plt.plot(episodes_list, data[0], label = 'alpha = 0.01')
    plt.plot(episodes_list, data[1], label = 'alpha = 0.1')
    plt.plot(episodes_list, data[2], label = 'alpha = 0.4')
    plt.grid(True)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(titulo)
    plt.xlim(0,1000)
    plt.legend()
    plt.savefig(plotFile+".jpg")     
    plt.close()

env = gym.make("Taxi-v3", render_mode='ansi').env

# Executa algorítimo de aprendizagem.
# Cria nova q_table
# Comentar caso queria utilizar tabela já existente
alpha = [0.01, 0.1, 0.4]
plot_list = []

for param in alpha:
    qlearn = QLearning(env, alpha=param, gamma=0.99, epsilon=0.7, epsilon_min=0.05, epsilon_dec=0.99, episodes=5000)
    q_table , rewards_list = qlearn.train()
    plot_list.append(rewards_list)

# Plotando gráfico com variação de parâmetro alpha:
plotactions('results/variable_alpha.jpg', plot_list, 'Episodes', 'Rewards Average Sum' , 'Rewards vs Episodes - Varying alpha hyperparameter')

# Descomentar para utilizar tabela criada
#q_table = loadtxt('data/q-table-taxi-driver.csv', delimiter=',')

# Utiliza a Q-table montada após treinamento para verificar performance do agente
'''
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

from IPython.display import clear_output
from time import sleep

clear_output(wait=True)

def print_frames(frames):
    for i, frame in enumerate(frames):
        clear_output(wait=True)
        print(frame['frame'])
        #print(frame['frame'].getvalue())
        print(f"Timestep: {i + 1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        sleep(.1)
        
print_frames(frames)

print("\n")
print("Timesteps taken: {}".format(epochs))
print("Penalties incurred: {}".format(penalties))
'''
