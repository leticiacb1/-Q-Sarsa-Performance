


def main():
    env = gym.make("CliffWalking-v0", render_mode="human").env

    (state, _) = env.reset()
    rewards = 0
    actions = 0
    done = False

    while not done:
        print(state)
        action = np.argmax(q_table[state])
        state, reward, done, truncated, info = env.step(action)

        rewards = rewards + reward
        actions = actions + 1

    print("\n")
    print("Actions taken: {}".format(actions))
    print("Rewards: {}".format(rewards)) 
