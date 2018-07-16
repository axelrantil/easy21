import numpy as np
from env import Env
from utils import epsilon_greedy, phi, get_q


def sarsa_control(N0, episodes, lambd, Q_opt):
    env = Env()
    N_s = np.zeros((11, 22))  # counter visited
    N_sa = np.zeros((2, 11, 22))  # count action taken at each state

    # Discount factor
    gamma = 1

    Q = np.zeros((2, 11, 22))

    mse = []

    # Episodes
    for k in range(0, episodes):
        E_sa = np.zeros((2, 11, 22))
        state = env.reset()  # Initiate state
        action = np.random.randint(0, 2)  # Initiate action

        # One episode/trajectory
        while True:
            state_tic, reward, terminal = env.step(action)

            # Update epsilon
            N_s[state[0], state[1]] += 1
            epsilon = N0 / (N0 + N_s[state[0], state[1]])

            action_tic = epsilon_greedy(Q, state_tic, epsilon)

            # Update delta/td error
            td_error = reward + gamma * Q[action_tic, state_tic[0], state_tic[1]] - Q[action, state[0], state[1]]
            E_sa[action, state[0], state[1]] += 1

            # Update alpha
            N_sa[action, state[0], state[1]] += 1
            alpha = 1 / N_sa[action, state[0], state[1]]

            Q += alpha * td_error * E_sa
            E_sa *= lambd*gamma

            state, action = state_tic, action_tic
            if terminal:
                break

        mse.append(np.mean((Q_opt - Q[:, 1:11, 1:22])**2))

    return Q[:, 1:11, 1:22], np.max(Q[:, 1:11, 1:22], axis=0), mse  # Q, V, mse


def sarsa_lfa_control(episodes, lambd, Q_opt):

    env = Env()
    gamma = 1  # Discount factor
    epsilon = 0.05  # Exploration factor
    alpha = 0.01  # Step-size

    Q = np.zeros((2, 11, 22))

    w = np.zeros((36,))

    mse = []

    # Episodes
    for k in range(0, episodes):

        E = np.zeros((36,))
        state = env.reset()  # Initiate state

        # One episode/trajectory
        while True:

            if np.random.rand() < epsilon:
                action = np.random.randint(0, 2)
            else:
                action = np.argmax([np.sum(phi(state, 0)*w),
                                    np.sum(phi(state, 1)*w)])

            state_tic, reward, terminal = env.step(action)

            if np.random.rand() < epsilon:
                action_tic = np.random.randint(0, 2)
            else:
                action_tic = np.argmax([np.sum(phi(state_tic, 0)*w),
                                        np.sum(phi(state_tic, 1)*w)])

            x = phi(state, action)
            x_tic = phi(state_tic, action_tic)

            Q_hat = np.sum(x * w)
            Q_hat_tic = np.sum(x_tic * w)

            td_error = reward + gamma * Q_hat_tic - Q_hat
            E = E * lambd * gamma + x
            delta_w = alpha * td_error * E

            w += delta_w

            state = state_tic

            if terminal:
                break

        Q = get_q(w)
        mse.append(np.mean((Q_opt - Q[:, 1:11, 1:22])**2))

    return Q[:, 1:11, 1:22], np.max(Q[:, 1:11, 1:22], axis=0), mse  # Q, V, mse
