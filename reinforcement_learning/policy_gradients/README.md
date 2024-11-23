
## Policy Gradients

# Policy Gradient Concepts

## **What is a Policy?**
A **policy** is a strategy used by an agent in reinforcement learning to decide which action to take in a given state. It can be represented in two forms:
- **Deterministic Policy**: A mapping directly from states to actions, denoted as \\( a = \\pi(s) \\).
- **Stochastic Policy**: A probability distribution over actions, denoted as \\( \\pi(a|s) \\), where the agent chooses actions based on probabilities.

In reinforcement learning, the policy is the core of the agent's decision-making process. The goal is to optimize this policy so that the agent maximizes cumulative rewards over time.

---

## **How to Calculate a Policy Gradient?**
The **policy gradient** is a technique used to optimize a policy by adjusting its parameters \\( \\theta \\) in the direction that maximizes expected rewards. The gradient of the expected return \\( J(\\theta) \\) with respect to \\( \\theta \\) is computed as:

\\[
\\nabla_\\theta J(\\theta) = \\mathbb{E}_{\\tau \\sim \\pi_\\theta} \\left[ \\sum_{t=0}^\\infty \\nabla_\\theta \\log \\pi_\\theta(a_t | s_t) \\cdot G_t \\right]
\\]

### Steps to calculate:
1. **Rollout**: Simulate episodes (trajectories \\( \\tau \\)) by following the current policy \\( \\pi_\\theta \\).
2. **Compute Returns**: For each episode, compute the **total return** \\( G_t \\), which is the sum of discounted rewards from time step \\( t \\) onward:
   \\[
   G_t = \\sum_{k=0}^\\infty \\gamma^k r_{t+k+1}
   \\]
   where \\( \\gamma \\) is the discount factor.
3. **Estimate Gradients**: Use the log likelihood trick to compute the gradient of the policy parameters:
   \\[
   \\nabla_\\theta J(\\theta) \\approx \\frac{1}{N} \\sum_{i=1}^N \\sum_{t=0}^{T_i} \\nabla_\\theta \\log \\pi_\\theta(a_t | s_t) \\cdot G_t
   \\]
   Here, \\( N \\) is the number of sampled episodes.

4. **Update Parameters**: Adjust the parameters \\( \\theta \\) using gradient ascent:
   \\[
   \\theta \\leftarrow \\theta + \\alpha \\nabla_\\theta J(\\theta)
   \\]
   where \\( \\alpha \\) is the learning rate.

---

## **What and How to Use a Monte-Carlo Policy Gradient?**
### **What is it?**
The **Monte-Carlo Policy Gradient** (also known as REINFORCE) is a method for policy optimization that uses **Monte-Carlo sampling** to estimate the policy gradient. It involves running episodes to completion and using the returns \\( G_t \\) from these episodes to update the policy.

### **How to Use it?**
1. **Initialize**: Start with a randomly initialized policy \\( \\pi_\\theta \\).
2. **Generate Episodes**: Interact with the environment to collect a batch of episodes by sampling actions using the current policy.
3. **Compute Returns**: For each time step in the episode, calculate the return \\( G_t \\).
4. **Update Policy**: Use the policy gradient formula to update the policy parameters:
   \\[
   \\theta \\leftarrow \\theta + \\alpha \\sum_{t=0}^{T-1} \\nabla_\\theta \\log \\pi_\\theta(a_t | s_t) \\cdot G_t
   \\]

### **Advantages**:
- Simple to implement.
- Does not require value function estimation.

### **Challenges**:
- High variance in gradient estimates.
- Slow convergence, especially in high-dimensional or sparse reward scenarios.

---

### Description
0. Simple Policy functionmandatoryWrite a functiondef policy(matrix, weight):that computes the policy with a weight of a matrix.$ cat 0-main.py
#!/usr/bin/env python3

import numpy as np
policy = __import__('policy_gradient').policy


weight = np.ndarray((4, 2), buffer=np.array([
    [4.17022005e-01, 7.20324493e-01],
    [1.14374817e-04, 3.02332573e-01],
    [1.46755891e-01, 9.23385948e-02],
    [1.86260211e-01, 3.45560727e-01]
    ]))
state = np.ndarray((1, 4), buffer=np.array([
    [-0.04428214,  0.01636746,  0.01196594, -0.03095031]
    ]))

res = policy(state, weight)
print(res)

$ ./0-main.py
[[0.50351642 0.49648358]]
$Repo:GitHub repository:holbertonschool-machine_learningDirectory:reinforcement_learning/policy_gradientsFile:policy_gradient.pyHelp×Students who are done with "0. Simple Policy function"Review your work×Correction of "0. Simple Policy function"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/12pts

1. Compute the Monte-Carlo policy gradientmandatoryBy using the previous function createdpolicy, write a functiondef policy_gradient(state, weight):that computes the Monte-Carlo policy gradient based on a state and a weight matrix.state: matrix representing the current observation of the environmentweight: matrix of random weightReturn: the action and the gradient (in this order)$ cat 1-main.py
#!/usr/bin/env python3

import gymnasium as gym
import numpy as np
import random
policy_gradient = __import__('policy_gradient').policy_gradient

def set_seed(env, seed=0):
    env.reset(seed=seed)
    np.random.seed(seed)
    random.seed(seed)

env = gym.make('CartPole-v1')
set_seed(env, 0)

weight = np.random.rand(4, 2)
state , _ = env.reset()
print(weight)
print(state)

action, grad = policy_gradient(state, weight)
print(action)
print(grad)

env.close()

$ ./1-main.py
[[0.5488135  0.71518937]
 [0.60276338 0.54488318]
 [0.4236548  0.64589411]
 [0.43758721 0.891773  ]]
[0.03132702 0.04127556 0.01066358 0.02294966]
1
[[-0.01554121  0.01554121]
 [-0.02047664  0.02047664]
 [-0.00529016  0.00529016]
 [-0.01138523  0.01138523]]
$Repo:GitHub repository:holbertonschool-machine_learningDirectory:reinforcement_learning/policy_gradientsFile:policy_gradient.pyHelp×Students who are done with "1. Compute the Monte-Carlo policy gradient"Review your work×Correction of "1. Compute the Monte-Carlo policy gradient"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/12pts

2. Implement the trainingmandatoryWrite a functiondef train(env, nb_episodes, alpha=0.000045, gamma=0.98):that implements a full training.env: initial environmentnb_episodes: number of episodes used for trainingalpha: the learning rategamma: the discount factorYou should usepolicy_gradient = __import__('policy_gradient').policy_gradientReturn: all values of the score (sum of all rewards during one episode loop)You need print the current episode number and the score after each loop in a format:Episode: {} Score: {}$ cat 2-main.py
#!/usr/bin/env python3

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import random
train = __import__('train').train

def set_seed(env, seed=0):
    env.reset(seed=seed)
    np.random.seed(seed)
    random.seed(seed)

env = gym.make('CartPole-v1')
set_seed(env, 0)

scores = train(env, 10000)

plt.plot(np.arange(len(scores)), scores)
plt.show()
env.close()

$ ./2-main.py
Episode: 0 Score: 22.0
Episode: 1 Score: 62.0
Episode: 2 Score: 48.0
Episode: 3 Score: 17.0
Episode: 4 Score: 30.0
Episode: 5 Score: 19.0
Episode: 6 Score: 19.0
Episode: 7 Score: 29.0
Episode: 8 Score: 28.0
Episode: 9 Score: 26.0
Episode: 10 Score: 24.0


....


Episode: 9990 Score: 500.0
Episode: 9991 Score: 371.0
Episode: 9992 Score: 500.0
Episode: 9993 Score: 500.0
Episode: 9994 Score: 500.0
Episode: 9995 Score: 500.0
Episode: 9996 Score: 500.0
Episode: 9997 Score: 500.0
Episode: 9998 Score: 500.0
Episode: 9999 Score: 500.0Note:we highly encourage you to modify the values ofalphaandgammato change the trend of the plotRepo:GitHub repository:holbertonschool-machine_learningDirectory:reinforcement_learning/policy_gradientsFile:train.pyHelp×Students who are done with "2. Implement the training"Review your work×Correction of "2. Implement the training"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/12pts

3. Animate iterationmandatoryIn the filetrain.py,  update the functiondef train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False)by adding a last optional parametershow_result(default:False).When this parameter is set toTrue, you should render the environment every 1000 episodes computed.$ cat 3-main.py
#!/usr/bin/env python3

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import random
train = __import__('train').train

def set_seed(env, seed=0):
    env.reset(seed=seed)
    np.random.seed(seed)
    random.seed(seed)

env = gym.make('CartPole-v1', render_mode="human")
set_seed(env, 0)

scores = train(env, 10000, 0.000045, 0.98, True)

env.close()

$ ./3-main.pyResult after few episodes:Result after more episodes:Result after 10000 episodes:Repo:GitHub repository:holbertonschool-machine_learningDirectory:reinforcement_learning/policy_gradientsFile:train.pyHelp×Students who are done with "3. Animate iteration"Review your work×Correction of "3. Animate iteration"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/7pts

**Repository:**
- GitHub repository: `holbertonschool-machine_learning`
- Directory: `supervised_learning/classification`
- File: `Policy_Gradients.md`
