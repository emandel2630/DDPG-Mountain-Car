import gym
import json
import numpy as np
import os
import os.path as osp
import time
import torch
import torch.nn.functional as F
import torch.optim as optim

from datetime import datetime

from DDPG import Actor, Critic
from OU_Noise import OU_Noise
from Replay_Buffer import Replay_Buffer
from utils import model_deep_copy

# Global Parameters
env_name = "MountainCarContinuous-v0"  # Environment name
use_cuda = True  # Flag to use CUDA for GPU acceleration
gpu_id = 0  # GPU ID

buffer_size = 1000000  # Size of the replay buffer
batch_size = 256  # Batch size for sampling from the replay buffer

mem_seed = 1  # Seed for random number generator in replay buffer
ou_seed = 1  # Seed for Ornstein-Uhlenbeck noise

lr_critic = 2e-2  # Learning rate for critic
lr_actor = 3e-3  # Learning rate for actor

episodes = 500  # Number of episodes to train

update_every_n_steps_DDPG =20
update_every_n_steps_TD3 = 20  # Frequency of updates per episode
learning_updates_per_learning_session_DDPG = 10  # Number of learning updates
learning_updates_per_learning_session_TD3 = 10

# Ornstein-Uhlenbeck noise parameters
mu = 0.0  # Mean
theta = 0.15  # Theta
sigma = 0.25  # Sigma

gamma = 0.99  # Discount factor for future rewards

critic_grad_clip_threshold = 5  # Gradient clipping for critic
actor_grad_clip_threshold = 5  # Gradient clipping for actor

tau_critic = 5e-3  # Soft update parameter for critic
tau_actor = 5e-3  # Soft update parameter for actor

win = 100  # Window size for calculating rolling score
score_th = 90  # Score threshold for winning

out_path = "mountain-car-master/Results"  # Directory to save results
if not osp.exists(out_path):
    os.makedirs(out_path)  # Create directory if it doesn't exist
    
def update_learning_rate(starting_lr, optimizer, rolling_score_list, score_th):
    """Dynamically adjust the learning rate based on performance."""
    # Adjust learning rate based on last rolling score
    if len(rolling_score_list) > 0:
        last_rolling_score = rolling_score_list[-1]
        # Adjust learning rate based on performance thresholds
        if last_rolling_score > 0.75 * score_th:
            new_lr = starting_lr / 100.0
        elif last_rolling_score > 0.6 * score_th:
            new_lr = starting_lr / 20.0
        elif last_rolling_score > 0.5 * score_th:
            new_lr = starting_lr / 10.0
        elif last_rolling_score > 0.25 * score_th:
            new_lr = starting_lr / 2.0
        else:
            new_lr = starting_lr
        # Apply new learning rate to all parameter groups
        for g in optimizer.param_groups:
            g['lr'] = new_lr

def select_action(state, actor_local, ou_noise, device):
    # Convert the current state into a tensor and move it to the specified device (CPU/GPU)
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    # Freeze the actor network to perform inference
    actor_local.eval()
    with torch.no_grad():
        # Predict an action from the current state
        action_numpy = actor_local(state).cpu().data.numpy().squeeze(0)
    # Re-enable training mode for the actor network
    actor_local.train()
    # Add noise to the action for exploration purposes
    action_numpy += ou_noise.sample()

    return action_numpy

def soft_update(local_model, target_model, tau):
    """Soft update model parameters."""
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


def train_DDPG(
    env, actor_local, actor_target, 
    critic_local, critic_target, 
    optim_actor, optim_critic, 
    replay_buffer, ou_noise, device
):
    
    """
    Train the agent using the DDPG algorithm.
    Parameters:
    - env: The environment to train in
    - actor_local: The local actor model
    - actor_target: The target actor model
    - critic_local: The local critic model
    - critic_target: The target critic model
    - optim_actor: The actor's optimizer
    - optim_critic: The critic's optimizer
    - replay_buffer: The replay buffer
    - ou_noise: Ornstein-Uhlenbeck noise for action exploration
    - device: The device to run the models on (CPU/GPU)
    """

    # Initialize counters and score trackers
    total_step_num = 0
    score_list = []
    rolling_score_list = []
    time_list = []
    max_score = float('-inf')
    max_rolling_score = float('-inf')
    
    # Main loop over episodes
    for i_episode in range(episodes):
        # Record the start time of the episode
        start = time.time()
        # Reset the environment and obtain the initial state
        state_numpy = env.reset()[0]
        # Initialize variables for the episode
        next_state_numpy = None
        action_numpy = None
        reward = None 
        done = False
        score = 0
        
        # Loop for each step of the episode
        while not done: 
            action_numpy= select_action(state_numpy, actor_local, ou_noise, device)
            
            # Perform the action in the environment to obtain the next state and reward
            next_state_numpy, reward, done, _,_ = env.step(action_numpy)
            # Accumulate the reward to the total score of the episode
            score += reward

            # Check if it's time to update the network (based on the number of steps)
            if len(replay_buffer) > batch_size and total_step_num % update_every_n_steps_DDPG == 0:
                # Perform several learning updates
                for _ in range(learning_updates_per_learning_session_DDPG):
                    # Sample a batch of experiences from the replay_buffer
                    states_numpy, actions_numpy, rewards_numpy, next_states_numpy, dones_numpy = replay_buffer.sample()
                    # Convert numpy arrays to tensors
                    states, actions, rewards, next_states, dones = [torch.from_numpy(x).float().to(device) for x in [states_numpy, actions_numpy, rewards_numpy, next_states_numpy, dones_numpy]]
                    dones = dones.unsqueeze(1)
                    
                    # Critic update
                    with torch.no_grad():
                        # Predict next actions and values using target networks
                        next_actions = actor_target(next_states)
                        next_value = critic_target(next_states, next_actions)

                        # Calculate target values for updating the critic
                        value_target = rewards + gamma * next_value * (1.0 - dones)

                    # Calculate expected values from the critic based on current states and actions
                    value = critic_local(states, actions)

                    # Compute loss using Mean Squared Error between target and expected values
                    loss_critic = F.mse_loss(value, value_target)
                    # Optimize the critic
                    optim_critic.zero_grad()
                    loss_critic.backward()

                    if critic_grad_clip_threshold is not None:
                        # Clip gradients to prevent exploding gradient problem
                        torch.nn.utils.clip_grad_norm_(critic_local.parameters(), critic_grad_clip_threshold)
                    optim_critic.step()
                    # Soft update of the target critic network parameters
                    soft_update(critic_local, critic_target, tau_critic)
                        
                    # Actor update
                    if done:
                        # Optionally adjust the actor's learning rate based on performance
                        update_learning_rate(lr_actor, optim_actor, rolling_score_list, score_th)
                    # Calculate loss for the actor
                    pred_actions = actor_local(states)
                    loss_actor = -critic_local(states, pred_actions).mean()
                    # Optimize the actor
                    optim_actor.zero_grad()
                    loss_actor.backward()
                    if actor_grad_clip_threshold is not None:
                        torch.nn.utils.clip_grad_norm_(actor_local.parameters(), actor_grad_clip_threshold)
                    optim_actor.step()
                    # Soft update of the target actor network parameters
                    soft_update(actor_local, actor_target, tau_actor)
            
            # Add the experience to the replay buffer
            replay_buffer.add_experience(state_numpy, action_numpy, reward, next_state_numpy, done)
                        # Update the current state with the next state for the next iteration
            state_numpy = next_state_numpy
            # Increment the global step index after each step of the episode
            total_step_num += 1
        
            
        # At the end of the episode, record the score and compute the rolling score
        score_list.append(score)
        rolling_score = np.mean(score_list[-win:])  # Calculate the rolling score over the last 'win' episodes
        rolling_score_list.append(rolling_score)

        # Update the maximum score and rolling score if the current episode's score is higher
        if score > max_score:
            max_score = score
            if i_episode >100:  # Start saving models after 100 episodes
                # Save the models with the highest score after 100 episodes
                torch.save(actor_local.state_dict(), osp.join(out_path, 'best_DDPG_actor.pth'))
                torch.save(critic_local.state_dict(), osp.join(out_path, 'best_DDPG_critic.pth'))

        if rolling_score > max_rolling_score:
            max_rolling_score = rolling_score
        
        # Calculate the time taken for the episode to complete
        end = time.time()
        time_list.append(end-start)

        # Print the episode's statistics including score, rolling score, max score, max rolling score, and time cost
        print(f"[Episode {i_episode:4d}: score: {score}; rolling score: {rolling_score}, max score: {max_score}, max rolling score: {max_rolling_score}, time cost: {end - start:.2f}]")
    
    # After training across all episodes, save the training results to a JSON file
    output = {
        "score_list": score_list, 
        "rolling_score_list": rolling_score_list, 
        "time_list": time_list,
        "max_score": max_score, 
        "max_rolling_score": max_rolling_score
    }
    json_name = osp.join(out_path, "DDPG.json")
    with open(json_name, 'w') as f:
        json.dump(output, f, indent=4)  # Save the dictionary as a JSON file with pretty formatting


def train_TD3(
    env, actor_local, actor_target, 
    critic_local, critic_target,
    critic_local_twin, critic_target_twin, 
    optim_actor, optim_critic, optim_critic_twin,
    replay_buffer, ou_noise, device
):
    
    """
    Train the agent using the TD3 algorithm.
    Parameters:
    - env: The environment to train in
    - actor_local: The local actor model
    - actor_target: The target actor model
    - critic_local: The local critic model
    - critic_target: The target critic model
    - critic_local_twin: The 2nd local critic model
    - critic_target_twin: The 2nd target critic model
    - optim_actor: The actor's optimizer
    - optim_critic: The critic's optimizer
    - optim_critic_twin: The optimizer for the twin critic
    - replay_buffer: The replay buffer
    - ou_noise: Ornstein-Uhlenbeck noise for action exploration
    - device: The device to run the models on (CPU/GPU)
    """

    # Initialize counters and score trackers
    total_step_num = 0
    score_list = []
    rolling_score_list = []
    time_list = []
    max_score = float('-inf')
    max_rolling_score = float('-inf')
    
    # Main loop over episodes
    for i_episode in range(episodes):
        # Record the start time of the episode
        start = time.time()
        # Reset the environment and obtain the initial state
        state_numpy = env.reset()[0]
        # Initialize variables for the episode
        next_state_numpy = None
        action_numpy = None
        reward = None 
        done = False
        score = 0
        
        # Loop for each step of the episode
        while not done: 
            action_numpy = select_action(state_numpy, actor_local, ou_noise, device)
            
            # Perform the action in the environment to obtain the next state and reward
            next_state_numpy, reward, done, _,_ = env.step(action_numpy)
            # Accumulate the reward to the total score of the episode
            score += reward

            # Check if it's time to update the network (based on the number of steps)
            if len(replay_buffer) > batch_size and total_step_num % update_every_n_steps_TD3 == 0:
                # Perform several learning updates
                for _ in range(learning_updates_per_learning_session_TD3):
                    # Sample a batch of experiences from the replay_buffer
                    states_numpy, actions_numpy, rewards_numpy, next_states_numpy, dones_numpy = replay_buffer.sample()
                    # Convert numpy arrays to tensors
                    states, actions, rewards, next_states, dones = [torch.from_numpy(x).float().to(device) for x in [states_numpy, actions_numpy, rewards_numpy, next_states_numpy, dones_numpy]]
                    dones = dones.unsqueeze(1)
                    
                    # Critic update
                    with torch.no_grad():
                        # Predict next actions and values using target networks
                        next_actions = actor_target(next_states)
                        noise = (torch.randn_like(actions) * 0.2).clamp(-0.5, 0.5)
                        next_actions = (next_actions + noise).clamp(-1.0,1.0)
                        next_value = critic_target(next_states, next_actions)
                        next_value_twin = critic_target_twin(next_states, next_actions)

                        # Take the minimum of the two critics for target Q value
                        next_value = torch.min(next_value, next_value_twin)

                        # Calculate target values for updating the critic
                        value_target = rewards + gamma * next_value * (1.0 - dones)

                    # Calculate expected values from the critic based on current states and actions
                    value = critic_local(states, actions)

                    # Update both critics
                    for critic, optim, in zip([critic_local, critic_local_twin], [optim_critic, optim_critic_twin]):
                        # Calculate expected values from the critic based on current states and actions
                        value = critic(states, actions)

                        # Compute loss using Mean Squared Error between target and expected values
                        loss_critic = F.mse_loss(value, value_target)
                        # Optimize the critic
                        optim.zero_grad()
                        loss_critic.backward()
                        if critic_grad_clip_threshold is not None:
                            torch.nn.utils.clip_grad_norm_(critic.parameters(), critic_grad_clip_threshold)
                        optim.step()

                    
                    # Delayed policy (actor) update
                    #if total_step_num % (update_every_n_steps_TD3 * 2) == 0:
                        # if done:
                        #     # Optionally adjust the actor's learning rate based on performance
                        #     update_learning_rate(lr_actor, optim_actor, rolling_score_list, score_th)

                    # Calculate loss for the actor
                    pred_actions = actor_local(states)
                    loss_actor = -critic_local(states, pred_actions).mean()
                    # Optimize the actor
                    optim_actor.zero_grad()
                    loss_actor.backward()
                    if actor_grad_clip_threshold is not None:
                        torch.nn.utils.clip_grad_norm_(actor_local.parameters(), actor_grad_clip_threshold)
                    optim_actor.step()

                    # Soft update of the target networks
                    soft_update(critic_local, critic_target, tau_critic)
                    soft_update(critic_local_twin, critic_target_twin, tau_critic)
                    soft_update(actor_local, actor_target, tau_actor)
            
            # Add the experience to the replay buffer
            replay_buffer.add_experience(state_numpy, action_numpy, reward, next_state_numpy, done)
                        # Update the current state with the next state for the next iteration
            state_numpy = next_state_numpy
            # Increment the global step index after each step of the episode
            total_step_num += 1
        
            
        # At the end of the episode, record the score and compute the rolling score
        score_list.append(score)
        rolling_score = np.mean(score_list[-win:])  # Calculate the rolling score over the last 'win' episodes
        rolling_score_list.append(rolling_score)

        # Update the maximum score and rolling score if the current episode's score is higher
        if score > max_score:
            max_score = score
            if i_episode >100:  # Start saving models after 100 episodes
                # Save the models with the highest score after 100 episodes
                torch.save(actor_local.state_dict(), osp.join(out_path, 'best_TD3_actor.pth'))
                torch.save(critic_local.state_dict(), osp.join(out_path, 'best_TD3_critic.pth'))
                torch.save(critic_local.state_dict(), osp.join(out_path, 'best_TD3_critic_twin.pth'))

        if rolling_score > max_rolling_score:
            max_rolling_score = rolling_score
        
        # Calculate the time taken for the episode to complete
        end = time.time()
        time_list.append(end-start)

        # Print the episode's statistics including score, rolling score, max score, max rolling score, and time cost
        print(f"[Episode {i_episode:4d}: score: {score}; rolling score: {rolling_score}, max score: {max_score}, max rolling score: {max_rolling_score}, time cost: {end - start:.2f}]")
    
    # After training across all episodes, save the training results to a JSON file
    output = {
        "score_list": score_list, 
        "rolling_score_list": rolling_score_list, 
        "time_list": time_list,
        "max_score": max_score, 
        "max_rolling_score": max_rolling_score
    }
    json_name = osp.join(out_path, "TD3.json")
    with open(json_name, 'w') as f:
        json.dump(output, f, indent=4)  # Save the dictionary as a JSON file with pretty formatting


def run_simulation(env, actor_model_path, device='cuda:0'):
    """
    Run a simulation using the best actor model.

    Parameters:
    - env_name: Name of the Gym environment to run the simulation in.
    - actor_model_path: Path to the saved best actor model.
    - device: Device to run the models on ('cpu' or 'cuda').
    """

    # Assuming the Actor class has been defined elsewhere and represents your actor model
    n_states = env.observation_space.shape[0]
    
    # Initialize the actor model
    actor = Actor(n_states).to(device)
    
    # Load the saved best actor model
    actor.load_state_dict(torch.load(actor_model_path, map_location=device))
    
    state = env.reset()[0]
    done = False
    score = 0

    start = time.time()
    # Run the simulation until the episode ends
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        # Set the actor to evaluation mode
        actor.eval()
        with torch.no_grad():
            # Get the action from the actor model
            action = actor(state).cpu().data.numpy().squeeze(0)
        
        # Perform the action
        next_state, reward, done, _, _ = env.step(action)
        score += reward
        state = next_state
    
    end = time.time()
    env.close()
    print(f"Simulation score: {score}, Time: {end-start}")


# """ Main Function"""
if __name__ == "__main__":
    env = gym.make(env_name, render_mode='human')
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    
    device = torch.device("cuda:%d" % gpu_id if use_cuda else "cpu")
    
    # critic 1
    critic_local = Critic(n_states, n_actions).to(device)
    critic_target = Critic(n_states, n_actions).to(device)
    model_deep_copy(from_model=critic_local, to_model=critic_target)
    
    optim_critic = optim.Adam(critic_local.parameters(), lr=lr_critic, eps=1e-4)

    #critic 2 (Only for TD3)
    critic_local_twin = Critic(n_states, n_actions).to(device)
    critic_target_twin = Critic(n_states, n_actions).to(device)
    model_deep_copy(from_model=critic_local_twin, to_model=critic_target_twin)
    
    optim_critic_twin = optim.Adam(critic_local_twin.parameters(), lr=lr_critic, eps=1e-4)
    
    replay_buffer = Replay_Buffer(buffer_size, batch_size, mem_seed)
    
    # actor
    actor_local = Actor(n_states).to(device)
    actor_target = Actor(n_states).to(device)
    model_deep_copy(from_model=actor_local, to_model=actor_target)
    
    optim_actor = optim.Adam(actor_local.parameters(), lr=lr_actor, eps=1e-4)
    
    # ou noise
    ou_noise = OU_Noise(
        size=n_actions, 
        seed=ou_seed,
        mu=mu,
        theta=theta, 
        sigma=sigma
    )
    ou_noise.reset()

    # train_TD3(
    #     env=env, 
    #     actor_local=actor_local, 
    #     actor_target=actor_target, 
    #     critic_local=critic_local, 
    #     critic_target=critic_target,
    #     critic_local_twin=critic_local_twin,
    #     critic_target_twin=critic_target_twin, 
    #     optim_actor=optim_actor, 
    #     optim_critic=optim_critic, 
    #     optim_critic_twin=optim_critic_twin,
    #     replay_buffer=replay_buffer, 
    #     ou_noise=ou_noise, 
    #     device=device
    # )
    
    #run_simulation(env,osp.join(out_path, 'best_TD3_actor.pth'))


    # print("DDPG")
    # train_DDPG(
    #     env=env, 
    #     actor_local=actor_local, 
    #     actor_target=actor_target, 
    #     critic_local=critic_local, 
    #     critic_target=critic_target, 
    #     optim_actor=optim_actor, 
    #     optim_critic=optim_critic, 
    #     replay_buffer=replay_buffer, 
    #     ou_noise=ou_noise, 
    #     device=device
    # )

    #run_simulation(env,osp.join(out_path, 'best_TD3_actor.pth'))