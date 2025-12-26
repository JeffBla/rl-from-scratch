import numpy as np
import gymnasium as gym
from collections import defaultdict

# Create env for training (no rendering)
train_env = gym.make("Blackjack-v1")


def argmax2(d):
    return 0 if d[0] > d[1] else 1


def target_policy(Q, state):
    return argmax2(Q[state])


def behavior_policy(Q, state, eps=0.2):
    greedy = argmax2(Q[state])

    a = None
    p = None
    if np.random.rand() < eps:
        a = np.random.choice([0, 1])
    else:
        a = greedy

    if a == greedy:
        p = 1 - eps + eps / 2.0
    else:
        p = eps / 2.0
    return a, p


def gen_episode(env, Q, policy):
    state, _ = env.reset()
    episode, done = [], False

    while not done:
        a, p = policy(Q, state)
        ns, r, terminated, truncated, _ = env.step(a)
        episode.append((state, a, r, p))
        state, done = ns, terminated or truncated
    return episode


def mc_policy_iter(env, alpha=0.1, gamma=1.0, num_episodes=10000, eps=0.1):
    Q = defaultdict(lambda: {0: 0.0, 1: 0.0})

    for i in range(num_episodes):
        episode = gen_episode(env, Q, behavior_policy)
        G, visited = 0.0, set()
        W = 1.0

        # Monte Carlo first-visit update
        for s, a, r, p in reversed(episode):
            G = gamma * G + r
            if (s, a) not in visited:
                visited.add((s, a))
                Q[s][a] += (W * G - Q[s][a]) * alpha

                W *= 1.0 / max(p, 1e-12)

    return Q


Q = mc_policy_iter(train_env, num_episodes=500000, eps=0.1)

# ----------- Evaluation with rendering -----------
eval_env = gym.make("Blackjack-v1", render_mode="human")


def run_step_by_step(env, policy, Q, episodes=1):
    for i in range(episodes):
        state, _ = env.reset()
        done = False
        print(f"\nEpisode {i+1} start: {state}")

        while not done:
            input("Press Enter to take next action...")
            a = policy(Q, state)
            print(f"State: {state}, Action: {'Stick' if a==0 else 'Hit'}")
            state, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated

        print("Final reward:", r)


run_step_by_step(eval_env, target_policy, Q, episodes=5)
