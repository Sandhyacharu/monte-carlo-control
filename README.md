# MONTE CARLO CONTROL ALGORITHM

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
Include the Monte Carlo control function

## OUTPUT:
![image](https://github.com/Sandhyacharu/monte-carlo-control/assets/75235167/a22c29d6-77de-4529-bee6-f40972d6b0e0)

![image](https://github.com/Sandhyacharu/monte-carlo-control/assets/75235167/22da95e2-0905-4d4f-8166-4f8131217746)

![image](https://github.com/Sandhyacharu/monte-carlo-control/assets/75235167/312a0d6d-b57c-441a-a7ab-75c6a2d4118f)

## RESULT:
Monte Carlo Control successfully learned an optimal policy for the specified environment.
