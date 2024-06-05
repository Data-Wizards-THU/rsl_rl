import tensorflow as tf
import numpy as np
import gym

# 创建环境
env = gym.make('CartPole-v1')
obs_space = env.observation_space.shape[0]
action_space = env.action_space.n

# 超参数
learning_rate = 0.0003
gamma = 0.99
clip_epsilon = 0.2
batch_size = 64
ppo_epochs = 10

# 策略网络
class PolicyNetwork(tf.keras.Model):
    def __init__(self, action_space):
        super(PolicyNetwork, self).__init__()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(128, activation='relu')
        self.logits = tf.keras.layers.Dense(action_space)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.logits(x)

# 值函数网络
class ValueNetwork(tf.keras.Model):
    def __init__(self):
        super(ValueNetwork, self).__init__()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(128, activation='relu')
        self.value = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.value(x)

policy_net = PolicyNetwork(action_space)
value_net = ValueNetwork()

policy_optimizer = tf.keras.optimizers.Adam(learning_rate)
value_optimizer = tf.keras.optimizers.Adam(learning_rate)

# 计算优势函数
def compute_advantages(rewards, values, gamma):
    advantages = []
    gae = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * (values[t + 1] if t + 1 < len(values) else 0) - values[t]
        gae = delta + gamma * gae
        advantages.insert(0, gae)
    return np.array(advantages)

# 更新策略网络
def update_policy(states, actions, advantages, old_log_probs):
    with tf.GradientTape() as tape:
        logits = policy_net(states)
        new_log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=actions, logits=logits)
        ratio = tf.exp(old_log_probs - new_log_probs)
        clipped_ratio = tf.clip_by_value(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
        loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))
    grads = tape.gradient(loss, policy_net.trainable_variables)
    policy_optimizer.apply_gradients(zip(grads, policy_net.trainable_variables))

# 更新值函数网络
def update_value(states, returns):
    with tf.GradientTape() as tape:
        values = value_net(states)
        loss = tf.reduce_mean(tf.square(returns - values))
    grads = tape.gradient(loss, value_net.trainable_variables)
    value_optimizer.apply_gradients(zip(grads, value_net.trainable_variables))

# 主训练循环
def train_ppo():
    for episode in range(1000):
        states = []
        actions = []
        rewards = []
        old_log_probs = []
        dones = []
        obs = env.reset()
        done = False
        total_reward = 0

        while not done:
            logits = policy_net(np.expand_dims(obs, axis=0))
            action = np.random.choice(action_space, p=tf.nn.softmax(logits).numpy()[0])
            log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=[action], logits=logits)
            next_obs, reward, done, _ = env.step(action)

            states.append(obs)
            actions.append(action)
            rewards.append(reward)
            old_log_probs.append(log_prob)
            dones.append(done)
            obs = next_obs
            total_reward += reward

        values = value_net(np.array(states)).numpy()
        advantages = compute_advantages(rewards, values, gamma)
        returns = advantages + values

        # 转换为数组
        states = np.array(states)
        actions = np.array(actions)
        old_log_probs = np.array(old_log_probs).squeeze()

        for _ in range(ppo_epochs):
            indices = np.random.permutation(len(states))
            for start in range(0, len(states), batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]

                update_policy(states[batch_indices], actions[batch_indices], advantages[batch_indices], old_log_probs[batch_indices])
                update_value(states[batch_indices], returns[batch_indices])

        print(f'Episode: {episode}, Total Reward: {total_reward}')

# 开始训练
train_ppo()
