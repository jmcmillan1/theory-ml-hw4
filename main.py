import numpy as np
import random
import matplotlib.pyplot as plt


def loose_follow(xvals, fns, eta):
    x_0, x_1 = xvals
    loss = [0, 0]
    total_loss = 0
    # for t in T
    for t in range(len(fns)):
        # summing to create the overall loss g_t(x_i)
        if t != 0:
            loss[0] += x_0*fns[t-1]
            loss[1] += x_1*fns[t-1]

        # with prob. choose an item
        prob = np.exp(-1*eta*loss[0])/(np.exp(-1*eta*loss[0]) + np.exp(-1*eta*loss[1]))
        if prob > random.uniform(0, 1):
            total_loss += x_0*fns[t]
        else:
            total_loss += x_1*fns[t]

    return total_loss


if __name__ == '__main__':
    x = [0, 1]
    T = 106
    functions = np.ones(T)
    for i in range(1, len(functions), 2):
        functions[i] = -2
    for i in range(2, len(functions)-1, 2):
        functions[i] = 2

    losses = np.zeros(104)
    for run in range(50):
        n = 1
        for i in range(1, 105, 1):
            losses[i-1] += loose_follow(x, functions, n)
            n = 1/(i * 200)

    print(losses/50)
    ks = [(i*200) for i in range(0, 104, 1)]
    ks[0] = 1
    plt.plot(ks, losses)
    plt.title("Loose follow the leader loss as a function of eta (50 trials)")
    plt.xlabel("K values")
    plt.ylabel("Loss values")
    plt.savefig("theory-ml-hw4")
