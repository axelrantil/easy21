import numpy as np
from env import Env
from utils import epsilon_greedy


def monte_carlo_control(N0, episodes):
    env = Env()

    N_s = np.zeros((11, 22))  # counter visited
    N_sa = np.zeros((2, 11, 22))  # count action taken at each state

    Q = np.zeros((2, 11, 22))

    # Episodes
    for k in range(0, episodes):
        trajectory = []
        state = env.reset()  # Initiate state
        trajectory.append(state) ## list() kan nog tas bort

        G = 0

        # One episode
        while True:
            # Update epsilon
            N_s[state[0], state[1]] += 1
            epsilon = N0 / (N0 + N_s[state[0], state[1]])

            action = epsilon_greedy(Q, state, epsilon)
            trajectory.append(action)

            # Update alpha
            N_sa[action, state[0], state[1]] += 1
            alpha = 1 / N_sa[action, state[0], state[1]]

            state, reward, terminal = env.step(action)

            G += reward  # No discount

            if terminal:
                break
            else:
                trajectory.append(state) ## list() kan nog tas bort?

        # Update all states
        for pos, event in enumerate(trajectory):
            if pos % 2 == 0:  # Action
                state = event
            else:  # State
                action = event
                Q[action, state[0], state[1]] += alpha * (G - Q[action, state[0], state[1]])

    return Q[:, 1:11, 1:22], np.max(Q[:, 1:11, 1:22], axis=0)  # Q, V


def main():
    Q, V = monte_carlo_control(N0=100, episodes=1000)  # Question 2
    print(V)


if __name__ == '__main__':
    main()