import gymnasium as gym

# actions: (Define action space)
# 0 = up
# 1 = right
# 2 = down
# 3 = left

# rewards:
# cada passo : -1 
# penhasco: -100

env = gym.make("CliffWalking-v0", render_mode='ansi').env

print("Action Space {}".format(env.action_space))
print("State Space {}".format(env.observation_space))
print('\n\n')

state = env.reset()
print(f"1 - Retorno reset:\n {state}\n")
print(f"2 - \n{env.render()}\n")

# escolhe uma acao aleatoria
action = env.action_space.sample()
print(f" Acao : {action}\n")

# executa a acao
state, reward, done, truncated, info = env.step(action)
print(f" Pos acao: {state}\n")
print(f" Estado: \n{env.render()}\n")
print(f" Reward: {reward}\n")

# executa a acao ir para north
state, reward, done, truncated, info = env.step(1)
print(f" Direita: \n{env.render()}\n")
print(f" Reward: {reward}\n")

state, reward, done, truncated, info = env.step(0)
print(f" Cima:\n{env.render()}\n")
print(f" Reward: {reward}\n")
