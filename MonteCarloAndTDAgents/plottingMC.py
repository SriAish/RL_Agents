from env import agent_by_mark
import matplotlib.pyplot as plt

def plot_mean_and_CI(mean, lb, ub, x, color_mean=None, color_shading=None):
    # plot the shaded range of the confidence intervals
    plt.fill_between(x, ub, lb,
                     color=color_shading, alpha=.2)
    # plot the mean on top
    plt.plot(x, mean, color_mean)
    
def plotting_MC(env, agent1, agent2, learn_fn, r=100):
    num_states = []
    num_roll_out = []
    ub = [[],[]]
    lb = [[],[]]
    mean = [[],[]]

    agent1.reset()
    # Training curves
    k = 0
    for j in range(r):
        num_roll_out.append(k)
        num_states.append(agent1.num_states())

        for i in range(15):
            R = 0
            for _ in range(10):
                state = env.reset()
                _, mark = state
                done = False
                agents = [agent1, agent2]
                while not done:
                    agent = agent_by_mark(agents, mark)
                    actions = env.available_actions()
                    action = agent.act(state, actions)
                    next_state, reward, done, _ = env.step(action)
                    _, mark = state = next_state

                R += reward
            if len(ub[0]) == j:
                    ub[0].append(R)
                    lb[0].append(R)
                    mean[0].append(R)
                    ub[1].append(-R)
                    lb[1].append(-R)
                    mean[1].append(-R)
            else:
                ub[0][j] = max(ub[0][j], R)
                lb[0][j] = min(lb[0][j], R)
                mean[0][j] = mean[0][j] + (R - mean[0][j])/(i+1)
                ub[1][j] = max(ub[1][j], -R)
                lb[1][j] = min(lb[1][j], -R)
                mean[1][j] = mean[1][j] + (-R - mean[1][j])/(i+1) 
        learn_fn(500)
        k+=500
    
    # plot the data
    fig = plt.figure(1, figsize=(14, 5))
    plot_mean_and_CI(mean[0], ub[0], lb[0], num_states, color_mean='r', color_shading='r')
    plot_mean_and_CI(mean[1], ub[1], lb[1], num_states, color_mean='b', color_shading='b')
    plt.title('Confidence interval vs states covered')
    plt.show()
    
    # plot the data
    fig = plt.figure(2, figsize=(14, 5))
    plot_mean_and_CI(mean[0], ub[0], lb[0], num_roll_out, color_mean='r', color_shading='r')
    plot_mean_and_CI(mean[1], ub[1], lb[1], num_roll_out, color_mean='b', color_shading='b')
    plt.title('Confidence interval vs number of rollouts')
    plt.show()