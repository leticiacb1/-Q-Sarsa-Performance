import matplotlib.pyplot as plt
from IPython.display import clear_output
from time import sleep

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
