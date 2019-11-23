import matplotlib.pyplot as plt

def plot_mean_and_CI(mean, lb, ub, x, color_mean=None, color_shading=None):
    # plot the shaded range of the confidence intervals
    plt.fill_between(x, ub, lb,
                     color=color_shading, alpha=.2)
    # plot the mean on top
    plt.plot(x, mean, color_mean)
    
def plotting(env, agent, learn_fn, r=100):    
    num_roll_out = []
    
    ub = []
    lb = []
    mean = []

    m = 0
    running_reward = 10
    for i in range(r):
        num_roll_out.append(m)
        for j in range(8):
            R = 0
            state = env.reset()
            steps = 0
            while steps < 2500:
                next_state, reward, done, _ = env.step(agent.select_action(state))
                R += reward
                if done:
                    break
                state = next_state
                steps+=1
            running_reward = 0.05 * R + (1 - 0.05) * running_reward
            if len(ub) == i:
                ub.append(R)
                lb.append(R)
                mean.append(R)
            else:
                ub[i] = max(ub[i], R)
                lb[i] = min(lb[i], R)
                mean[i] = mean[i] + (R - mean[i])/(j+1)
        if running_reward > env.spec.reward_threshold:
            break
        agent = learn_fn(env, agent, 1)
        m+=1
    
    # plot the data
    fig = plt.figure(1, figsize=(14, 5))
    plot_mean_and_CI(mean, ub, lb, num_roll_out, color_mean='g', color_shading='g')
    plt.show()
    
    return agent

def plotting_pendulum(env, agent, learn_fn, r=100):    
    num_roll_out = []
    
    ub = []
    lb = []
    mean = []

    m = 0
    for i in range(r):
        num_roll_out.append(m)
        for j in range(8):
            R = 0
            state = env.reset()
            steps = 0
            while steps < 2500:
                next_state, reward, done, _ = env.step(agent.select_action(state))
                R += reward
                if done:
                    break
                state = next_state
                steps+=1
            if len(ub) == i:
                ub.append(R)
                lb.append(R)
                mean.append(R)
            else:
                ub[i] = max(ub[i], R)
                lb[i] = min(lb[i], R)
                mean[i] = mean[i] + (R - mean[i])/(j+1)
                
        agent = learn_fn(env, agent, 1)
        m+=1
    
    # plot the data
    fig = plt.figure(1, figsize=(14, 5))
    plot_mean_and_CI(mean, ub, lb, num_roll_out, color_mean='g', color_shading='g')
    plt.show()
    
    return agent