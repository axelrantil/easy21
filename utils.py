import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def add_card(state, add_to_dealer=False):
    """ Adds a card to the state

    Keyword arguments:
    state -- state list
    dealer -- True if dealer gets card otherwise False (default False)
    """
    color = np.random.rand()
    if add_to_dealer:
        if color < 0.33333:
            state[0] -= np.random.randint(1, 11)
        else:
            state[0] += np.random.randint(1, 11)
    else:
        if color < 0.33333:
            state[1] -= np.random.randint(1, 11)
        else:
            state[1] += np.random.randint(1, 11)
    return state


def epsilon_greedy(Q, s, eps):
    r = np.random.rand()
    if r < eps:
        return np.random.randint(0, 2)
    else:
        return np.argmax(Q[:, s[0], s[1]])


def phi(s, a):
    features = np.zeros((3, 6, 2))

    idx1 = [1 <= s[0] <= 4, 4 <= s[0] <= 7, 7 <= s[0] <= 10]
    idx2 = [1 <= s[1] <= 6, 4 <= s[1] <= 9, 7 <= s[1] <= 12,
            10 <= s[1] <= 15, 13 <= s[1] <= 18, 16 <= s[1] <= 21]

    features[idx1, idx2, a] = 1

    return features.flatten()


def get_q(w):
    Q = np.zeros((2, 11, 22))
    for s0 in range(1, 11):
        for s1 in range(1, 22):
            for a in range(2):
                Q[a, s0, s1] = np.sum(phi([s0, s1], a)*w)
    return Q


def plot_V(V):
    dn, pn = 21, 10
    d = range(0, dn, 1)
    p = range(0, pn, 1)
    D, P = np.meshgrid(d, p)
    ptic = [str(i + 1) for i in p]
    dtic = [str(i + 1) if i % 3 == 2 else '' for i in d]

    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')
    plt.xticks(p, ptic)
    plt.yticks(d, dtic)
    ha.set_xlabel('Dealer card showing')
    ha.set_ylabel('Player card sum')
    ha.set_zlabel('Value')
    ha.plot_surface(P, D, V, cmap=cm.coolwarm)

    plt.show()