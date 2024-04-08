# DDPG-Mountain-Car
DDPG and TD3 approaches to solving the continuous mountain car problem 

The Twin Delayed Deep Deterministic Policy Gradient (TD3) and Deep Deterministic Policy Gradient (DDPG) algorithms represent significant advancements in the field of Reinforcement Learning, particularly in the domain of continuous action spaces. These algorithms extend the principles of value-based and policy-based methods, aiming to combine the benefits of both to achieve efficient and stable learning in environments where actions are not discrete but rather continuous.
DDPG is similar to DQN in that its basis is the Q-function and it is off-policy. The difference is its intended usage in continuous action spaces. In DQN the Q values are determined for each action and the maximum is select from that set as the proper choice. In a continuous action space this is impossible as there are an infinite number of Q values and solving the optimization problem in a traditional way would be very computationally expensive. Since the Q function is differential with respect to the action a gradient-based learning rule can be used to approximate the max Q value.

 ![image](https://github.com/emandel2630/DDPG-Mountain-Car/assets/91342800/77ec4f8b-4964-4ee3-a2dc-0f00d6340cbd)

Fig 1. Control flow of DDPG: https://arshren.medium.com/step-by-step-guide-to-implementing-ddpg-reinforcement-learning-in-pytorch-9732f42faac9
	One of the key features of both DDPG and TD3 is the usage of target networks as shown in the diagram above. The target network lags the original network and is updated occasionally by simply copying the original network. When trying to minimize loss the original and target networks are compared and due to the time delay the loss minimization is kept from becoming unstable.
	Lastly, without any noise in the environment the agent cannot explore the entirety of the action space. Time correlated OU noise is typically used as it doesn’t diverge too far from the mean and is temporally correlated which makes it more similar to the continuous environments of the real world. This type of noise is a better approximation of real-world conditions and can be tuned very precisely to fit the environment making it a clear choice.

TD3, as an extension of DDPG, introduces key modifications to address the overestimation bias inherent in value-based methods and to improve the stability and performance of the learning process. TD3 distinguishes itself with three key enhancements:
1.	Clipped Double-Q Learning: Utilizing two Critic networks and taking the minimum value to estimate the target Q-value reduces overestimation bias.
2.	Policy Delay: Delays policy updates to ensure the value function has time to stabilize, reducing variance in policy evaluation.
3.	Target Policy Smoothing: Adds noise to the target action, smoothing out the value landscape and preventing the exploitation of sharp value estimations.

 ![image](https://github.com/emandel2630/DDPG-Mountain-Car/assets/91342800/ebccc397-d986-4e45-9f64-fd59f45401ca)

Fig 2. Control flow of TD3 (https://www.researchgate.net/figure/Structure-of-TD3-Twin-Delayed-Deep-Deterministic-Policy-Gradient-with-RAMDP_fig2_338605159)


![image](https://github.com/emandel2630/DDPG-Mountain-Car/assets/91342800/d073535e-3d44-4ee2-9547-ebfc99577145)
![image](https://github.com/emandel2630/DDPG-Mountain-Car/assets/91342800/5674fdf2-a600-4052-93b4-600d48b64cc6)

  
Fig 3. Pseudocode of DDPG and TD3 (https://spinningup.openai.com/)
	As shown above the two algorithms are very similar. The action clipping on line 12, the policy delay on line 15, and presence of two critic networks in TD3 are fundamentally the only differences between DDPG and TD3. These few changes build upon the DDPG algorithm in an attempt to overcome its weaknesses, and as a result TD3 outperforms DDPG on a variety of continuous action space tasks with faster convergence rates and overall better performance.

Before I display the results, I have a few notes on my training methodology:
Dynamic Learning Rate Adjustment
I implemented a dynamic learning rate adjustment mechanism based on the performance of the agent, measured against a predefined score threshold. This strategy ensures that the learning rate is finely tuned in response to the agent's progress, facilitating faster convergence during phases of high performance and encouraging exploration during lower performance phases.

Exploration with Ornstein-Uhlenbeck Noise
To encourage exploration, I incorporated Ornstein-Uhlenbeck (OU) noise in the action selection process. This choice was motivated by the OU noise's temporal correlation properties, making it ideal for environments where actions have a continuous nature. The parameters of the OU noise (mu, theta, sigma) were carefully chosen to balance exploration and exploitation, ensuring that the agent does not settle prematurely on suboptimal policies.

Gradient Clipping
To combat the potential instability due to large gradients, I employed gradient clipping for both the actor and critic networks. This precautionary measure ensures that updates remain manageable, preventing the model parameters from diverging or oscillating excessively.

Soft Updates for Model Parameters
The use of soft updates (controlled by the tau parameter) for the target networks promotes stability in the learning process. By gradually incorporating the learned weights into the target models, the training process benefits from smoother transitions and reduced volatility in the policy and value estimations.

Experience Replay Buffer
A large experience replay buffer was utilized to store and reuse past experiences, mitigating the issues associated with correlated data. By sampling from this buffer, the models learn from a diverse set of experiences, improving generalization and efficiency. The chosen buffer size and batch size were optimized based on computational resources and the complexity of the task.

Hyper Parameters:
	For both DDPG and TD3 I had nearly identical hyperparameters. All of the following hyperparameters listed will be the setting chose for both methods unless explicitly stated. 
•	Buffer Size = 1000000
•	Batch Size = 256
•	Learning rate (actor) = 0.003
•	Learning rate (critic(s)) = 0.02
•	Number of episodes = 500
•	Number of steps between updates = 20
•	Updates per learning session = 10
•	OU Noise parameters
o	Mu = 0 
o	Theta = 0.15
o	Sigma = 0.25
•	Gamma = 0.99
•	Gradient Clipping threshold (TD3 only) = 5
•	Polyak Learning Rate = 0.005
•	Winning score value = 90
These values are the values used in “Addressing Function Approximation Error in Actor-Critic Methods” by Fujimoto et. Al (https://arxiv.org/abs/1802.09477) and they worked very well. 
The resulting score and time plots are below:
![image](https://github.com/emandel2630/DDPG-Mountain-Car/assets/91342800/add18498-920d-48ec-ad80-d837d9d987b4)

 
Fig 4. DDPG vs TD3 scores 

 ![image](https://github.com/emandel2630/DDPG-Mountain-Car/assets/91342800/7583b6c4-a6ab-4131-ab69-b0450133900e)

Fig 5. DDPG vs TD3 time to solution 
	The results above show a clear convergence of the models. They both approach that 90 point threshold in terms of score while having competition times converge to ~2.5 seconds. TD3 and DDPG converged at the same rate with a relatively stable state being reached by 120 epochs. TD3 had greater variance between trials most likely as a result of the two critics yielding a greater uncertainty of outcome. DDPG slightly outperformed TD3 in terms of score by epoich 500 but not significantly. DDPG and TD3 both brought the solution time down to the same ~2.5 seconds. 
	TD3 is typically supposed to outperform DDPG due to its twin critics preventing overfitting while simultaneously converging more quickly. Perhaps due to the hyperparameters being fine tuned for DDPG it was given the edge over TD3. Another observation of note was how delaying the actor update in TD3 significantly reduced accuracy and convergence. I had to lower the delay to be in synch with the critic updates for TD3 to converge at all which was unusual. 
	Both TD3 and DDPG are both effective offline approaches for tackling the Mountain-Car problem and problems similar to it. 

Acknowledgments:
I extend my gratitude to the open-source community and specifically to the GitHub user "greatwallet" for providing a solid foundation and inspiration for this project's implementation.
