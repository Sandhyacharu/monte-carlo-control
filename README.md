### EX NO: 05
### DATE:
# <p align="center">MONTE CARLO CONTROL ALGORITHM</p>

## AIM
The aim of Monte Carlo Control is to develop an optimal policy for a Markov Decision Process (MDP) by estimating the value function and improving the policy through repeated sampling and evaluation of episodes. This technique is used in reinforcement learning to find the best actions to take in each state in order to maximize cumulative rewards.

## PROBLEM STATEMENT
Monte Carlo Control is a reinforcement learning method, to figure out the best actions for different situations in an environment. The provided code is meant to do this, but it's currently having issues with variables and functions.

## MONTE CARLO CONTROL ALGORITHM
### Step 1:
Initialize Q-values, state-value function, and the policy.

### Step 2:
Interact with the environment to collect episodes using the current policy.

### Step 3:
For each time step within episodes, calculate returns (cumulative rewards) and update Q-values.

### Step 4:
Update the policy based on the improved Q-values.

### Step 5:
Repeat steps 2-4 for a specified number of episodes or until convergence.

### Step 6:
Return the optimal Q-values, state-value function, and policy.

## MONTE CARLO CONTROL FUNCTION
```python3
# Developed by: Sandhya Charu N
# Register Number: 212220230041

import numpy as np
from collections import defaultdict

def mc_control(env, gamma=1.0, init_alpha=0.5, min_alpha=0.01, alpha_decay_ratio=0.5,
               init_epsilon=1.0, min_epsilon=0.1, epsilon_decay_ratio=0.9,
               n_episodes=3000, max_steps=200, first_visit=True):

    nS, nA = env.observation_space.n, env.action_space.n
    Q = defaultdict(lambda: np.zeros(nA))
    V = defaultdict(float)
    pi = defaultdict(lambda: np.random.choice(nA))
    Q_track = []
    pi_track = []

    select_action = lambda state,Q, epsilon: np.argmax(Q[state]) if np.random.random() > epsilon else np.random.randint(len(Q[state]))

    for episode in range(n_episodes):
        epsilon = max(init_epsilon * (epsilon_decay_ratio ** episode), min_epsilon)
        alpha = max(init_alpha * (alpha_decay_ratio ** episode), min_alpha)
        trajectory = generate_trajectory(select_action, Q, epsilon, env, max_steps)
        n = len(trajectory)
        G = 0
        for t in range(n - 1, -1, -1):
            state, action, reward, _, _ = trajectory[t]
            G = gamma * G + reward
            if first_visit and (state, action) not in [(s, a) for s, a, _, _, _ in trajectory[:t]]:
                Q[state][action] += alpha * (G - Q[state][action])
                V[state] = np.max(Q[state])
                pi[state] = np.argmax(Q[state])
        Q_track.append(Q.copy())
        pi_track.append(pi.copy)
    return Q, V, pi

optimal_Q, optimal_V, optimal_pi = mc_control(env)
print_state_value_function(optimal_Q, P,n_cols=4, prec=2, title='Action-value-function')
print_state_value_function(optimal_Q, P,n_cols=4, prec=2, title='State-value-function')
print_policy(optimal_pi,P)
```

## OUTPUT:
![image](https://github.com/Sandhyacharu/monte-carlo-control/assets/75235167/a22c29d6-77de-4529-bee6-f40972d6b0e0)

![image](https://github.com/Sandhyacharu/monte-carlo-control/assets/75235167/22da95e2-0905-4d4f-8166-4f8131217746)

![image](https://github.com/Sandhyacharu/monte-carlo-control/assets/75235167/312a0d6d-b57c-441a-a7ab-75c6a2d4118f)

## RESULT:
Monte Carlo Control successfully learned an optimal policy for the specified environment.
