import matplotlib.pyplot as plt
import tensorflow as tf
from dqn import preprocess_state
import numpy as np
import sys

def plot_mean_and_CI(mean, lb, ub, x, color_mean=None, color_shading=None):
    # plot the shaded range of the confidence intervals
    plt.fill_between(x, ub, lb,
                     color=color_shading, alpha=.2)
    # plot the mean on top
    plt.plot(x, mean, color_mean)
    
def plotting_DQN(env, agent, learn_fn, r=100):    
    num_roll_out = []
    
    ub = []
    lb = []
    mean = []

    val = []

    m = 0
    for i in range(r):
        print(i, end="")
        sys.stdout.flush()
        agent, G, v = learn_fn(env, agent, 5)
        m+=5
        num_roll_out.append(m)
        ub.append(np.max(G))
        lb.append(np.min(G))
        mean.append(np.mean(G))
        val.append(v)
    
    # plot the data
    fig = plt.figure(1, figsize=(14, 5))
    plot_mean_and_CI(mean, ub, lb, num_roll_out, color_mean='g', color_shading='g')
    plt.show()

    plt.plot(num_roll_out, val)
    plt.show()
    
    return agent
