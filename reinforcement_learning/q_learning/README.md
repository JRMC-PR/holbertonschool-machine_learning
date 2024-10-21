
## Q-learning
# Reinforcement Learning Concepts

## What is a Markov Decision Process (MDP)?
A **Markov Decision Process** is a mathematical framework used to model decision-making in environments where outcomes are partly random and partly under the control of the agent. It is defined by:
- **S**: A set of states.
- **A**: A set of actions.
- **P**: A transition probability matrix, which defines the probability of moving from one state to another given an action.
- **R**: A reward function, which gives the immediate reward received after transitioning from one state to another due to an action.
- **γ (gamma)**: A discount factor that determines the importance of future rewards.

MDPs provide a structured way to model environments for reinforcement learning tasks.

---

## What is an environment?
The **environment** is the external system with which an agent interacts. It provides:
- **States**: Observations or situations the agent perceives.
- **Actions**: The set of all possible actions an agent can take.
- **Rewards**: Feedback on the agent’s actions, which helps guide learning.

The environment defines the rules of the interaction and evolves based on the actions the agent takes.

---

## What is an agent?
An **agent** is the learner or decision-maker in a reinforcement learning problem. It interacts with the environment by:
- Observing the **state** of the environment.
- Taking **actions** based on a **policy**.
- Receiving **rewards** to learn and improve its policy over time.

The goal of the agent is to maximize cumulative rewards over time by learning an optimal policy.

---

## What is a state?
A **state** represents the current situation of the environment. It contains all the information needed for the agent to make a decision. States can be fully observable (where the agent knows everything about the environment) or partially observable.

---

## What is a policy function?
A **policy function** (π) defines how an agent behaves by mapping states to actions. It can be:
- **Deterministic**: Always selecting the same action for a given state.
- **Stochastic**: Selecting an action based on a probability distribution over possible actions for a given state.

In reinforcement learning, the agent’s goal is to learn an optimal policy that maximizes cumulative reward.

---

## What is a value function? A state-value function? An action-value function?
- **Value Function (V)**: Measures the expected cumulative reward an agent can achieve starting from a particular state.
- **State-Value Function (V(s))**: The expected reward starting from state **s** and following a certain policy.
- **Action-Value Function (Q(s, a))**: The expected reward starting from state **s**, taking action **a**, and following a certain policy thereafter.

These functions help the agent estimate the long-term benefit of states and actions.

---

## What is a discount factor?
The **discount factor** (γ) is a number between 0 and 1 that determines how much future rewards are valued compared to immediate rewards. A high discount factor (close to 1) means future rewards are considered almost as important as immediate rewards, while a low discount factor means the agent focuses more on short-term rewards.

---

## What is the Bellman equation?
The **Bellman equation** provides a recursive decomposition of the value function. It relates the value of a state to the values of possible next states by considering both immediate rewards and future rewards:
\[ V(s) = R(s, a) + \gamma \sum_{s'} P(s'|s, a) V(s') \]
This equation is key to understanding how to optimize policies in dynamic environments.

---

## What is epsilon greedy?
**Epsilon greedy** is a strategy for balancing exploration and exploitation in reinforcement learning. With probability ε, the agent explores by selecting a random action, and with probability **1 - ε**, the agent exploits by selecting the action with the highest estimated value.

---

## What is Q-learning?
**Q-learning** is a **value-based** reinforcement learning algorithm that learns the **Q-value function**. It estimates the quality of actions in a given state and seeks to find the optimal policy by iteratively updating the Q-values using the Bellman equation:
\[ Q(s, a) \leftarrow Q(s, a) + \alpha [R(s, a) + \gamma \max_{a'} Q(s', a') - Q(s, a)] \]
Q-learning is an off-policy algorithm, meaning the agent can learn the optimal policy while following another (possibly random) policy.

---


### Description
Question #0What is reinforcement learning?A type of supervised learning, because the rewards supervise the learningA type of unsupervised learning, because there are no labels for each actionIts own subcategory of machine learning

Question #1What is an environment?The place in which actions can be performedA description of what the agent seesA list of actions that can be performedA description of which actions the agent should perform

Question #2An agent chooses its action based on:The current stateThe value functionThe policy functionThe previous reward

Question #3What is a policy function?A description of how the agent should be rewardedA description of how the agent should behaveA description of how the agent could be rewarded in the futureA function that is learnedA function that is set at the beginning

Question #4What is a value function?A description of how the agent should be rewardedA description of how the agent should behaveA description of how the agent could be rewarded in the futureA function that is learnedA function that is set at the beginning

Question #5What is epsilon-greedy?A type of policy functionA type of value functionA way to balance policy and value functionsA balance exploration and exploitation

Question #6What is Q-learning?A reinforcement learning algorithmA deep reinforcement learning algorithmA value-based learning algorithmA policy-based learning algorithmA model-based approach

0. Load the EnvironmentmandatoryWrite a functiondef load_frozen_lake(desc=None, map_name=None, is_slippery=False):that loads the pre-madeFrozenLakeEnvevnironment fromgymnasium:descis eitherNoneor a list of lists containing a custom description of the map to load for the environmentmap_nameis eitherNoneor a string containing the pre-made map to loadNote: If bothdescandmap_nameareNone, the environment will load a randomly generated 8x8 mapis_slipperyis a boolean to determine if the ice is slipperyReturns: the environment$ cat 0-main.py
#!/usr/bin/env python3

load_frozen_lake = __import__('0-load_env').load_frozen_lake

env = load_frozen_lake()
print(env.unwrapped.desc)
print(len(env.unwrapped.P[0][0]))
print(env.unwrapped.P[0][0])

env = load_frozen_lake(is_slippery=True)
print(env.unwrapped.desc)
print(len(env.unwrapped.P[0][0]))
print(env.unwrapped.P[0][0])

desc = [['S', 'F', 'F'], ['F', 'H', 'H'], ['F', 'F', 'G']]
env = load_frozen_lake(desc=desc)
print(env.unwrapped.desc)

env = load_frozen_lake(map_name='4x4')
print(env.unwrapped.desc)
$ ./0-main.py
[[b'S' b'F' b'F' b'F' b'F' b'F' b'F' b'F']
 [b'H' b'F' b'H' b'F' b'F' b'H' b'F' b'F']
 [b'F' b'F' b'H' b'H' b'F' b'F' b'H' b'F']
 [b'F' b'F' b'F' b'F' b'F' b'H' b'F' b'H']
 [b'F' b'H' b'F' b'F' b'F' b'F' b'F' b'F']
 [b'H' b'F' b'F' b'F' b'F' b'F' b'F' b'H']
 [b'F' b'H' b'F' b'F' b'F' b'H' b'F' b'F']
 [b'F' b'F' b'F' b'F' b'F' b'H' b'F' b'G']]
1
[(1.0, 0, 0.0, False)]
[[b'S' b'F' b'F' b'F' b'F' b'F' b'H' b'H']
 [b'F' b'F' b'H' b'F' b'H' b'F' b'F' b'H']
 [b'H' b'F' b'H' b'F' b'F' b'F' b'F' b'F']
 [b'F' b'H' b'F' b'F' b'F' b'F' b'F' b'F']
 [b'F' b'F' b'H' b'F' b'F' b'F' b'F' b'H']
 [b'F' b'F' b'H' b'F' b'F' b'F' b'F' b'F']
 [b'F' b'F' b'F' b'H' b'H' b'F' b'F' b'F']
 [b'F' b'F' b'F' b'F' b'F' b'H' b'F' b'G']]
3
[(0.3333333333333333, 0, 0.0, False), (0.3333333333333333, 0, 0.0, False), (0.3333333333333333, 8, 0.0, False)]
[[b'S' b'F' b'F']
 [b'F' b'H' b'H']
 [b'F' b'F' b'G']]
[[b'S' b'F' b'F' b'F']
 [b'F' b'H' b'F' b'H']
 [b'F' b'F' b'F' b'H']
 [b'H' b'F' b'F' b'G']]
$Repo:GitHub repository:holbertonschool-machine_learningDirectory:reinforcement_learning/q_learningFile:0-load_env.pyHelp×Students who are done with "0. Load the Environment"Review your work×Correction of "0. Load the Environment"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/7pts

1. Initialize Q-tablemandatoryWrite a functiondef q_init(env):that initializes the Q-table:envis theFrozenLakeEnvinstanceReturns: the Q-table as anumpy.ndarrayof zeros$ cat 1-main.py
#!/usr/bin/env python3

load_frozen_lake = __import__('0-load_env').load_frozen_lake
q_init = __import__('1-q_init').q_init

env = load_frozen_lake()
Q = q_init(env)
print(Q.shape)
env = load_frozen_lake(is_slippery=True)
Q = q_init(env)
print(Q.shape)
desc = [['S', 'F', 'F'], ['F', 'H', 'H'], ['F', 'F', 'G']]
env = load_frozen_lake(desc=desc)
Q = q_init(env)
print(Q.shape)
env = load_frozen_lake(map_name='4x4')
Q = q_init(env)
print(Q.shape)
$ ./1-main.py
(64, 4)
(64, 4)
(9, 4)
(16, 4)
$Repo:GitHub repository:holbertonschool-machine_learningDirectory:reinforcement_learning/q_learningFile:1-q_init.pyHelp×Students who are done with "1. Initialize Q-table"Review your work×Correction of "1. Initialize Q-table"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/12pts

2. Epsilon GreedymandatoryWrite a functiondef epsilon_greedy(Q, state, epsilon):that uses epsilon-greedy to determine the next action:Qis anumpy.ndarraycontaining the q-tablestateis the current stateepsilonis the epsilon to use for the calculationYou should samplepwithnumpy.random.uniformnto determine if your algorithm should explore or exploitIf exploring, you should pick the next action withnumpy.random.randintfrom all possible actionsReturns: the next action index$ cat 2-main.py
#!/usr/bin/env python3

load_frozen_lake = __import__('0-load_env').load_frozen_lake
q_init = __import__('1-q_init').q_init
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy
import numpy as np

desc = [['S', 'F', 'F'], ['F', 'H', 'H'], ['F', 'F', 'G']]
env = load_frozen_lake(desc=desc)
Q = q_init(env)
Q[7] = np.array([0.5, 0.7, 1, -1])
np.random.seed(0)
print(epsilon_greedy(Q, 7, 0.5))
np.random.seed(1)
print(epsilon_greedy(Q, 7, 0.5))
$ ./2-main.py
2
0
$Repo:GitHub repository:holbertonschool-machine_learningDirectory:reinforcement_learning/q_learningFile:2-epsilon_greedy.pyHelp×Students who are done with "2. Epsilon Greedy"Review your work×Correction of "2. Epsilon Greedy"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/12pts

3. Q-learningmandatoryWrite the functiondef train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):that performs Q-learning:envis theFrozenLakeEnvinstanceQis anumpy.ndarraycontaining the Q-tableepisodesis the total number of episodes to train overmax_stepsis the maximum number of steps per episodealphais the learning rategammais the discount rateepsilonis the initial threshold for epsilon greedymin_epsilonis the minimum value thatepsilonshould decay toepsilon_decayis the decay rate for updatingepsilonbetween episodesWhen the agent falls in a hole, the reward should be updated to be-1You should useepsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedyReturns:Q, total_rewardsQis the updated Q-tabletotal_rewardsis a list containing the rewards per episode$ cat 3-main.py
#!/usr/bin/env python3

load_frozen_lake = __import__('0-load_env').load_frozen_lake
q_init = __import__('1-q_init').q_init
train = __import__('3-q_learning').train
import numpy as np

np.random.seed(0)
desc = [['S', 'F', 'F'], ['F', 'H', 'H'], ['F', 'F', 'G']]
env = load_frozen_lake(desc=desc)
Q = q_init(env)

Q, total_rewards  = train(env, Q)
print(Q)
split_rewards = np.split(np.array(total_rewards), 10)
for i, rewards in enumerate(split_rewards):
    print((i+1) * 500, ':', np.mean(rewards))
$ ./3-main.py
[[0.96059595 0.970299   0.95098641 0.96059508]
 [0.96059578 0.         0.00914612 0.43792863]
 [0.17824547 0.         0.         0.        ]
 [0.97029808 0.9801     0.         0.96059035]
 [0.         0.         0.         0.        ]
 [0.         0.         0.         0.        ]
 [0.9800999  0.98009991 0.99       0.97029895]
 [0.98009936 0.98999805 1.         0.        ]
 [0.         0.         0.         0.        ]]
500 : 0.918
1000 : 0.962
1500 : 0.948
2000 : 0.946
2500 : 0.948
3000 : 0.964
3500 : 0.95
4000 : 0.934
4500 : 0.928
5000 : 0.934
$Note : The output may vary based on your implemntation but for every run, you should have the same resultRepo:GitHub repository:holbertonschool-machine_learningDirectory:reinforcement_learning/q_learningFile:3-q_learning.pyHelp×Students who are done with "3. Q-learning"0/16pts

4. PlaymandatoryWrite a functiondef play(env, Q, max_steps=100):that has the trained agent play an episode:envis theFrozenLakeEnvinstanceQis anumpy.ndarraycontaining the Q-tablemax_stepsis the maximum number of steps in the episodeYou need to update0-load_env.pyto addrender_mode="ansi"Each state of the board should be displayed via the consoleYou should always exploit the Q-tableEnsure that the final state of the environment is also displayed after the episode concludes.Returns: The total rewards for the episode and a list of rendered outputs representing the board state at each step.$ cat 4-main.py
#!/usr/bin/env python3

load_frozen_lake = __import__('0-load_env').load_frozen_lake
q_init = __import__('1-q_init').q_init
train = __import__('3-q_learning').train
play = __import__('4-play').play

import numpy as np

np.random.seed(0)
desc = [['S', 'F', 'F'], ['F', 'H', 'H'], ['F', 'F', 'G']]
env = load_frozen_lake(desc=desc)
Q = q_init(env)

Q, _ = train(env, Q)

env.reset()
total_rewards, rendered_outputs = play(env, Q)

print(f'Total Rewards: {total_rewards}')
for output in rendered_outputs:
    print(output)
$ ./4-main.py
Total Rewards: 1.0

`S`FF
FHH
FFG
  (Down)
SFF
`F`HH
FFG
  (Down)
SFF
FHH
`F`FG
  (Right)
SFF
FHH
F`F`G
  (Right)
SFF
FHH
FF`G`

$Repo:GitHub repository:holbertonschool-machine_learningDirectory:reinforcement_learning/q_learningFile:4-play.pyHelp×Students who are done with "4. Play"Review your work×Correction of "4. Play"Start a new testCloseRequirement successRequirement failCode successCode failEfficiency successEfficiency failText answer successText answer failSkipped - Previous check failed0/7pts

**Repository:**
- GitHub repository: `holbertonschool-machine_learning`
- Directory: `supervised_learning/classification`
- File: `Qlearning.md`
