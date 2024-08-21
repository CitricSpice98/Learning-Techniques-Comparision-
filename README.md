# Learning-Techniques-Comparision-


Objectives

 Open AI Gym provides a standard set of Environments and customizable environments to train, evaluate, and compare the performance of various Reinforcement learning techniques. The main objective of the given assignment is to compare the efficiency and performance of the DQN Reinforcement Learning technique based on Deterministic policies evaluated on environments with both discrete and continuous observation spaces, against the Policy Gradient Techniques such as Deep Deterministic Policy Gradient (DDPG) and Soft Actor Critic (SAC).
The primary goal is to identify, design, and evaluate the optimal Reinforcement learning techniques to solve the problem identified in the selected game environment. Successful learning in Open AI Gym environments demonstrates an algorithm's ability to generalize its learned policies across, which is crucial for real-world applications. Experimenting with hyper-parameters like learning rate, discount factors, timesteps, and other factors allows us to obtain optimal solutions.
Description of the Task
Taking an objective look at Task 1 and Task 2, the first step is to identify the Environments and Reinforcement Learning Techniques that fit the requirements of the Assignment. Once the Environments are identified, we then identify the Reinforcement Learning Techniques t be evaluated. For Task-1, we will consider Deep Q Network as mentioned in the task breakdown whereas for Task-2 we will consider the Policy Gradient Techniques like DDPG and SAC. We then train the model and tune the hyperparameters till the point of convergence. We then obtain and compare the reward plots and determine the optimal technique based on efficiency and accuracy.
 
Details of the experiment.
The Details of the experiment deals with the step-by-step approach to drawing our conclusions, including the discussion of the configurations of environments , models, and hyperparameters. The first step is to identify and install the dependencies and packages required for the assignment. This is followed by setting up the environments, designing and tuning the models, and then comparing the reward functions.
Environments
The first environment we discuss is the Taxi-v3 environment, with discrete action and observation spaces. In this environment, the agent controls a taxi navigating the grid world and ferrying passengers to their destination. There are 6 action spaces for the agent in the environment
• ‘0’ – Move South
• ‘1’ – Move North
• ‘2’ – Move East
• ‘3’ – Move West
• ‘4’ – Pick up Passenger.
• ‘5’ – Drop Passenger
The Discrete Observation space for the environment is a 500x5 grid with 25 values for the position of the car (5x5) grid and 5 ( 4 for the location of the passengers Drop off locations and 1 for the location of passengers to be picked up.
The second environment we discuss is the CartPole – v1 environment, which involves balancing a pole on a moving cart .
The action space for the environment is discrete with 2 states
• ‘0’ to move cart left
• ‘1’ to move cart right
The observation space for the environment is continuous with 4 spaces 1) Cart Position – Horizontal Position of the cart. Range [-4.8, 4.8]
2) Cart Velocity – velocity of the cart, Unbounded.
 
3) Pole Angle – Angle of the pole with respect to vertical axes in radians. Range[- 0.418,0.418].
4) Pole Angular Velocity – The angular velocity of the pole, Unbounded.
The Third environment we discuss is the continuous action and observation spaces of Pendulum-v1, which we will use to evaluate and compare both policy Gradient Techniques. The pendulum problem deals with trying to balance a swinging pendulum.
The action space for this environment is continuous, an applied torque to the joint of the pendulum in the range [-2.0,2.0] units.
The observation space for this environment is also continuous, with 3 parameters
1. Cosineoftheangle–Thecosineoftheanglebetweenthependulumand
vertical axes in the Range [-1.0, 1.0]
2. Sine of the angle - The sine of the angle between the pendulum and
vertical axes in the Range[-1.0,1.0]
3. AngularVelocity–Theangularvelocityofthependulum,Unbounded.
Reinforcement Learning techniques
DQN or Deep Q Networks, is a reinforcement learning technique that combines deep neural networks with Q learning, enabling agents to learn optimal policies in complex environments. Q learning is a model-free reinforcement learning algorithm that computes a cumulative reward for the action space based on the target states. DQN incorporates the computation of the Q function through a Deep Neural Network which accepts the input states as input and gives the Q function as output. This helps break the temporal correlation and improves the stability of learning.
The Algorithm for DQN –
1. InitializetheQnetwork
2. Interact with the environment to obtain the observation space 3. ComputetheTDerrorandupdatetheQfunction
4. PeriodicallyupdatetheTargetnetwork

For Task 1 of the Assignment, we will be using the stable baselines implementation of DQN, using the pre-designed network to train and evaluate the model in the Taxi-v3 and CartPole-v1 environments.
DDPG or Deep Deterministic Policy Gradient Techniques, is a reinforcement learning technique designed for continuous action spaces. It combines the DQN technique for discrete action spaces with policy gradients for continuous action spaces. DDPG aims to learn a deterministic policy, meaning that for a given state, the policy outputs a specific action rather than a probability distribution of actions.
The DDPG favors an Actor-Critic Architecture with 2 neural networks-
• Actor-Network - The actor is responsible for learning the policy. It takes the
space as the input and gives a deterministic action as the output.
• Critic Network – The critic evaluates the Q-value of the state action pair.
The actor loss aims to maximize the Q- value, while the critic loss aims to minimize the TD error.
The Algorithm for DDPG –
1. Initializeactor,critic,target-actor,andtarget-criticandexperiencereplay buffer.
2. Interactwiththeenvironment,collectexperiences,andstoretheminthe replay buffer.
3. Training
• Sample a batch of experiences from the replay buffer.
• Update the critic network to minimize the TD error.
• Update the actor network to maximize the expected Q-value.
• Update the target networks.
4. Repeatsteps2&3untilconvergence.
Similar to Task1, in Task2 the environment will be trained and evaluated on a pre- designed implementation of the DDPG technique on stable baselines.
SAC or the Soft Actors Critic is an advanced reinforcement learning algorithm designed to train agents in continuous action spaces. It combines ideas from actor-critic methods, maximum entropy reinforcement learning, and the soft Q- learning framework. SAC aims to learn both an optimal policy and optimal value function in a stable and efficient manner. SAC introduces the concept of entropy

to be stochastic and explores action space more thoroughly. Entropy is the measure of uncertainty in the system, which SAC tries to maximize. SAC learns both the value function and Q function. The value function estimates the expected return from a state, while the Q function estimates the expected return from taking a specific action in a state. It updates the actor by considering both the expected return (Q-value) and the entropy of the policy. SAC maintains 2 Q functions which reduces overestimation bias. The minimum of the 2 is selected to be the target for value function and actor updates.
The Algorithm for SAC –
1. Initialization:initializetheactor,2Q-functions,targetq-function,value
function, target value function and experience buffer.
2. ExperienceCollection:interactwiththeenvironment,collectexperiences,
and store them in the replay buffer.
3. Training:Sampleabatchoftheexperiencesfromthereplaybuffer.Update
the Q-function, Value-function, and the actor.
The implementation of the SAC for Task 2 is done by using the pre-trained SAC model provided by stable baselines.
There is also a custom evaluation policy method to generate the average and episodic reward functions.

Results and Conclusions
TASK 1
Following the training and evaluation using Custom Evaluation policy method, we get the following episodic reward functions.
Discrete Observation Space
Continuous Observation Space
   
DQN is highly effective on discrete observation spaces , while having a modified or tuned version of the DQN works better on continuous action spaces.

Task 2
DDPG for Pendulum-v1
 SAC for Pendulum-v1

 From the episodic reward plots of the 2 reinforcement learning techniques applied on the pendulum-v1 environment, we can see that both algorithms converge to a mean reward of around (– 120), Which indicates that the performance of both algorithms to solve the problem is quite similar.
SAC tends to be more efficient for exploration problems due to the incorporation of entropy regularization. It is also more sample-efficient since the inclusion of entropy in the objective function can help the agent learn more quickly from collected experiences. It is also more stable during training since the use of 2 q functions contributes to training stability, reducing the risk of divergence or getting stuck in sub-optimal policies. It is suited for continuous control tasks.
DDPG learns a deterministic policy which is advantageous in situations where there is a clear, deterministic action is desired for a given state. It can suffer from training instability. Careful hyperparameter tuning is crucial to achieve stable and efficient learning.
Since the task at hand – solving the pendulum problem ie finding the exact torque to apply to stop the momentum of the pendulum, is more suited to the DDPG technique due to the state of the hyperparameters and the presence of a clear action space.

References :
1) https://stable-baselines3.readthedocs.io/en/master/common/envs.html 2)https://github.com/openai/gym/blob/master/gym/envs/classic_control/acrobo t.py
3) https://github.com/msachin93/RL/blob/master/CartPole/cartpole.ipynb
4) https://medium.com/intro-to-artificial-intelligence/soft-actor-critic- reinforcement-learning-algorithm-1934a2c3087f 5)https://spinningup.openai.com/en/latest/algorithms/ddpg.html
6) https://chat.openai.com/
