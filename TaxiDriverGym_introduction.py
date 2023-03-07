import gymnasium as gym

env = gym.make("Taxi-v3", render_mode='ansi').env

print("Action Space {}".format(env.action_space))
print("State Space {}".format(env.observation_space))
print('\n\n')

state = env.reset()
print(f"1 - Retorno reset:\n {state}\n")
print(f"2 - {env.render()}")

# escolhe uma acao aleatoria
action = env.action_space.sample()
print(f" Acao : {action}")

# executa a acao
state, reward, done, truncated, info = env.step(action)
print(f"Pos acao: {state}\n")
print(f"Estado: {env.render()}\n")
print(f"Reward: {reward}\n")

# executa a acao ir para north
state, reward, done, truncated, info = env.step(1)
print(f" Norte: \n{env.render()}\n")
print(f"Reward: {reward}\n")

state, reward, done, truncated, info = env.step(0)
print(f"Sul :\n{env.render()}")
print(f"Reward: {reward}\n")

# actions: (Define action space)
# 0 = south
# 1 = north
# 2 = east
# 3 = west
# 4 = pickup
# 5 = dropoff

# Quantos espaços possíveis o ambiente Taxi-v3 possui?
#   -> observation_space == todos os possíveis estados  

# Quantas ações o agente que atua no ambiente Taxi-v3 possui? 
#   -> 6 AÇÕES

# O que a variável reward retornada por env.step(<number>) significa?
#   ->Execute um passo de tempo da dinâmica do ambiente usando as ações do agente.
#     Recompensa do resultado da ação.

env.close()
