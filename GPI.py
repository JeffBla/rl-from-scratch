import numpy as np
import gymnasium as gym

env = gym.make('Taxi-v3', render_mode="human")


def policy_evaluation(gramma, policy: np.array, theta, V):
    while True:
        delta = 0
        for s in range(env.observation_space.n):
            v = 0
            for action, action_prob in enumerate(policy[s]):
                for prob, next_s, reward, done in env.P[s][action]:
                    if done:
                        v += action_prob * prob * reward
                    else:
                        v += action_prob * prob * (reward + gramma * V[next_s])
            delta = max(delta, np.abs(v - V[s]))
            V[s] = v
        if delta < theta:
            break


def policy_improvement(gamma, policy, V):
    policy_stable = True

    for s in range(env.observation_space.n):
        old_policy = policy[s].copy()
        q_s = np.zeros(env.action_space.n)
        for a in range(env.action_space.n):
            for prob, next_s, reward, done in env.P[s][a]:
                if done:
                    q_s[a] += prob * reward
                else:
                    q_s[a] += prob * (reward + gamma * V[next_s])
        best_action = np.argmax(q_s)
        policy[s] = np.eye(env.action_space.n)[best_action]
    if np.any(policy != old_policy):
        policy_stable = False

    return policy_stable


def policy_iteration(gamma, max_iteration, theta):
    V = np.zeros(env.observation_space.n)
    policy = np.ones([env.observation_space.n, env.action_space.n
                      ]) / env.action_space.n
    for i in range(max_iteration):
        policy_evaluation(gamma, policy, theta, V)
        stable = policy_improvement(gamma, policy, V)
        if stable:
            break
    return V, policy


V, policy = policy_iteration(gamma=0.99, max_iteration=10000, theta=0.0001)

observation, info = env.reset()

episode_over = False
total_reward = 0
step = 0
state = 0

while not episode_over:
    env.render()
    action = np.argmax(policy[state])

    next_state, reward, terminated, truncated, info = env.step(action)

    total_reward += reward
    state = next_state
    step += 1
    episode_over = terminated or truncated

print(f'total reward: {total_reward}, steps: {step}')
