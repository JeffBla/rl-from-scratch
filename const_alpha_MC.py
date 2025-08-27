import numpy as np
import gymnasium as gym
from collections import defaultdict

# Create env for training (no rendering)
train_env = gym.make("Blackjack-v1")


def gen_episode(env, policy):
    state, _ = env.reset()
    episode, done = [], False

    while not done:
        a = policy(state)
        ns, r, terminated, truncated, _ = env.step(a)
        episode.append((state, a, r))
        state, done = ns, terminated or truncated
    return episode


def mc_policy_iter(env, alpha=0.1, gamma=1.0, num_episodes=10000, eps=0.1):
    Q = defaultdict(lambda: {0: 0.0, 1: 0.0})

    # initial random policy
    def policy(s):
        return np.random.choice([0, 1])

    for i in range(num_episodes):
        episode = gen_episode(env, policy)
        G, visited = 0.0, set()

        # Monte Carlo first-visit update
        for s, a, r in reversed(episode):
            G = gamma * G + r
            if (s, a) not in visited:
                visited.add((s, a))
                Q[s][a] += (G - Q[s][a]) * alpha

        # Policy improvement: ε-greedy
        def policy(s, Q=Q, eps=eps):
            if np.random.rand() < eps:
                return np.random.choice([0, 1])
            return 0 if Q[s][0] >= Q[s][1] else 1

    # final greedy policy
    def greedy_policy(s):
        return 0 if Q[s][0] >= Q[s][1] else 1

    return Q, greedy_policy


Q, pi = mc_policy_iter(train_env, num_episodes=500000, eps=0.1)

# ----------- Evaluation with rendering -----------
eval_env = gym.make("Blackjack-v1", render_mode="human")


def run_step_by_step(env, policy, episodes=1):
    for i in range(episodes):
        state, _ = env.reset()
        done = False
        print(f"\nEpisode {i+1} start: {state}")

        while not done:
            input("Press Enter to take next action...")
            a = policy(state)
            print(f"State: {state}, Action: {'Stick' if a==0 else 'Hit'}")
            state, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated

        print("Final reward:", r)


run_step_by_step(eval_env, pi, episodes=5)
