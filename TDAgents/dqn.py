import numpy as np
import gym
import keras
import random
import itertools
import sys, os
import tensorflow as tf
from keras.models import Sequential
from keras.models import Model, load_model
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Flatten
from keras.layers import Dense, multiply, Input, Activation
from keras.models import model_from_json
from keras import losses
from keras.optimizers import RMSprop

global_step = tf.Variable(0, name='global_step', trainable=False)
updates = 0
episode_t = 0
first = True
states_g = []

def preprocess_state(state):
    state = state[35:195:2, ::2]
    state = np.mean(state, axis=2).astype(np.uint8)
    return state
    
class Estimator():
    def __init__(self, scope="estimator", summaries_dir=None):
        self.scope = scope
        # Writes Tensorboard summaries to disk
        self.summary_writer = None
        with tf.compat.v1.variable_scope(scope):
            # Build the graph
            self._build_model()
            if summaries_dir:
                summary_dir = os.path.join(summaries_dir, "summaries_{}".format(scope))
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
                self.summary_writer = tf.summary.FileWriter(summary_dir)

    def _build_model(self):
        with tf.compat.v1.variable_scope(self.scope, reuse=tf.AUTO_REUSE) as scope:
            # Placeholders for our input
            # Our input are 4 grayscale frames of shape 84, 84 each
            self.X_pl = tf.compat.v1.placeholder(shape=[None, 80, 80, 4], dtype=tf.uint8, name="X")
            # The TD target value
            self.y_pl = tf.compat.v1.placeholder(shape=[None], dtype=tf.float32, name="y")
            # Integer id of which action was selected
            self.actions_pl = tf.compat.v1.placeholder(shape=[None], dtype=tf.int32, name="actions")

            X = tf.to_float(self.X_pl) / 255.0
            batch_size = tf.shape(self.X_pl)[0]

            # Three convolutional layers
            conv1 = tf.contrib.layers.conv2d(
                X, 32, 8, 4, activation_fn=tf.nn.relu)
            conv2 = tf.contrib.layers.conv2d(
                conv1, 64, 4, 2, activation_fn=tf.nn.relu)
            conv3 = tf.contrib.layers.conv2d(
                conv2, 64, 3, 1, activation_fn=tf.nn.relu)

            # Fully connected layers
            flattened = tf.contrib.layers.flatten(conv3)
            fc1 = tf.contrib.layers.fully_connected(flattened, 512)
            self.predictions = tf.contrib.layers.fully_connected(fc1, 4)

            # Get the predictions for the chosen actions only
            gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_pl
            self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)

            # Calculate the loss
            self.losses = tf.squared_difference(self.y_pl, self.action_predictions)
            self.loss = tf.reduce_mean(self.losses)

            # Optimizer Parameters from original paper
            self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
            self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())

            # Summaries for Tensorboard
            self.summaries = tf.summary.merge([
                tf.summary.scalar("loss", self.loss),
                tf.summary.histogram("loss_hist", self.losses),
                tf.summary.histogram("q_values_hist", self.predictions),
                tf.summary.scalar("max_q_value", tf.reduce_max(self.predictions))
            ])


    def predict(self, sess, s):
        return sess.run(self.predictions, { self.X_pl: s })

    def update(self, sess, s, a, y):
        feed_dict = { self.X_pl: s, self.y_pl: y, self.actions_pl: a }
        summaries, global_step, _, loss = sess.run(
            [self.summaries, tf.contrib.framework.get_global_step(), self.train_op, self.loss],
            feed_dict)
        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)
        return loss

class DQN():
    replay_memory = []
    def __init__(self, model, target_model, num_actions, discount_factor=0.99):
        self.model = model
        self.target_model = target_model
        self.discount_factor = discount_factor
        self.actions = np.reshape(np.ones(num_actions), [1, num_actions])
        self.num_actions = num_actions
        self.replay_memory_size = 33
        self.batch_size = 32

    def copy_model(self, sess):
        e1_params = [t for t in tf.compat.v1.trainable_variables() if t.name.startswith(self.model.scope)]
        e1_params = sorted(e1_params, key=lambda v: v.name)
        e2_params = [t for t in tf.compat.v1.trainable_variables() if t.name.startswith(self.target_model.scope)]
        e2_params = sorted(e2_params, key=lambda v: v.name)

        update_ops = []
        for e1_v, e2_v in zip(e1_params, e2_params):
            op = e2_v.assign(e1_v)
            update_ops.append(op)

        sess.run(update_ops)
        
    def act(self, sess, state):
        action_values = self.model.predict(sess, state)
        best_action = np.argmax(action_values)
        return best_action

    def epsilon_greedy_action(self, sess, state, epsilon):
        if random.random() <= epsilon:
            action = random.randrange(self.num_actions)
            return action
        return self.act(sess, state)

    def update(self, states, actions, rewards, next_states, done, sess):
        # First, predict the Q values of the next states. Note how we are passing ones as the mask.
        q_values_next = self.target_model.predict(sess, next_states)
        # The Q values of each start state is the reward + gamma * the max next state Q value
        targets = rewards + np.invert(done).astype(np.float32) * self.discount_factor * np.max(q_values_next, axis=1)
        # Fit the keras model. Note how we are passing the actions as the mask and multiplying
        # the targets by the actions.
        loss = self.model.update(sess, states, actions, targets)
        return loss 

def learn(sess, env, agent, num_episodes):
    global updates, episode_t
    # Create directories for checkpoints and summaries
    experiment_dir = os.path.abspath("./experiments/{}".format(env.spec.id))
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints_try")
    checkpoint_path = os.path.join(checkpoint_dir, "model")
    monitor_path = os.path.join(experiment_dir, "monitor")

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.compat.v1.train.Saver()
    # Load a previous checkpoint if we find one
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        saver.restore(sess, latest_checkpoint)

    epsilons = np.linspace(1, 0.01, 1000)
    state = env.reset()
    state = preprocess_state(state)
    state = np.stack([state] * 4, axis=2)
    if len(DQN.replay_memory) == 0:
        for i in range(agent.replay_memory_size):
            action = agent.epsilon_greedy_action(sess, state.reshape(1, 80, 80, 4), epsilons[episode_t])
            next_state, reward, done, _ = env.step(action)
            next_state = preprocess_state(next_state)
            next_state = np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis=2)
            DQN.replay_memory.append((state, action, reward, next_state, done))
            if done:
                state = env.reset()
                state = preprocess_state(state)
                state = np.stack([state] * 4, axis=2)
            else:
                state = next_state
    epsilon = 0.01
    for episode in range(num_episodes):
        episode_t = episode_t + 1
        saver.save(tf.compat.v1.get_default_session(), checkpoint_path)

        if episode_t < 1000:
            epsilon = epsilons[episode_t]
        state = env.reset()
        state = preprocess_state(state)
        state = np.stack([state] * 4, axis=2)
        loss = None

        for t in itertools.count():
            updates += 1
            if updates % 500 == 0:
                agent.copy_model(sess)

            action = agent.epsilon_greedy_action(sess, state.reshape(1, 80, 80, 4), epsilon)
            next_state, reward, done, _ = env.step(action)
            next_state = preprocess_state(next_state)
            next_state = np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis=2)
            if len(DQN.replay_memory) == agent.replay_memory_size:
                DQN.replay_memory.pop(0)

            DQN.replay_memory.append((state, action, reward, next_state, done))
            
            samples = random.sample(DQN.replay_memory, agent.batch_size)
            
            states, actions, rewards, next_states, dones = map(np.array, zip(*samples))
            
            loss = agent.update(states, actions, rewards, next_states, dones, sess)

            if done or t > 2500:
                break

            state = next_state

    rew = np.arange(5)
    for i in range(5):
        state = env.reset()
        state = preprocess_state(state)
        state = np.stack([state] * 4, axis=2)
        loss = None
        R = 0
        for t in itertools.count():
            sys.stdout.flush()
            action = agent.act(sess, state.reshape(1, 80, 80, 4))
            next_state, reward, done, _ = env.step(action)
            next_state = preprocess_state(next_state)
            next_state = np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis=2)

            R += reward
            if done or t > 2500:
                break

            state = next_state
        rew[i] = R

    val = 0
    vals = agent.model.predict(sess, states_g)
    val = np.mean(vals)
    return agent, rew, val

def learn_DQN(env, agent, num_episodes):
    global first
    if first:
        print("populating states")
        state = env.reset()
        state = preprocess_state(state)
        state = np.stack([state] * 4, axis=2)
        done = False
        while not done:
            states_g.append(state)
            action = random.randrange(agent.num_actions)
            next_state, reward, done, _ = env.step(action)
            next_state = preprocess_state(next_state)
            next_state = np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis=2)
            state = next_state
        first = False
    with tf.compat.v1.Session() as sess:
        sess.run(tf.global_variables_initializer())
        return learn(sess, env, agent, num_episodes)
