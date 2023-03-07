import matplotlib.pyplot as plt
from IPython.display import clear_output
from time import sleep

def plot_compare_algorithms(q_rewards_list, sarsa_rewards_list, episodes, filename):
    # Montar gráfico de comparação entre algorítimos:
    episodes_list = range(0, episodes, 10) 
    plt.plot(episodes_list, q_rewards_list, label = 'Qlearning')
    plt.plot(episodes_list, sarsa_rewards_list , label = 'Sarsa')
    plt.grid(True)
    plt.xlabel('Episodes')
    plt.ylabel('Average sum of reward')
    plt.title('Comparing Algorithms: Qlearning vs Sarsa')
    plt.xlim(0,1000)
    plt.legend()
    plt.savefig('results/'+ filename+'_comparing_algorithms'+".jpg")     
    plt.close()

def print_frames(frames):
    for i, frame in enumerate(frames):
        clear_output(wait=True)
        print(f"\n--------- Timestep - {i + 1} ----------\n")
        #print(frame['frame'])
        #print(frame['frame'].getvalue())
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}\n")
        sleep(.1)
