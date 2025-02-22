
import gymnasium as gym
import numpy as np
from itertools import count
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import time

GAMMA = 1.0 #0.99
LR = 0.01
N_EPISODES = 2000
N_ITRS_PER_EPISODE = 200


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 32)
        self.affine2 = nn.Linear(32, 2)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affine1(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)

    def act(self, state) -> tuple[int, float]:
        state = torch.from_numpy(state).float().unsqueeze(0)

        # predict the probability distribution
        probs = self(state)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)



def train():

  

    env = gym.make('CartPole-v1') #, render_mode="human")
    env.reset(seed=42)
    torch.manual_seed(42)

    policy = Policy()
    optimizer = optim.Adam(policy.parameters(), lr=LR)


    scores_deque = deque(maxlen=100)
    for i_episode in range(N_EPISODES):

        # for every episode reset the environment
        state, _ = env.reset()
        ep_reward = 0

        rewards = []
        saved_log_probs = []
        itr_counter = 0
        for t in range(1, N_ITRS_PER_EPISODE): 

            # select action 
            action, log_prob = policy.act(state)

            # step in the environment
            state, reward, done, _, _ = env.step(action)
            #env.render()
            
            ep_reward += reward

            rewards.append(reward)
            saved_log_probs.append(log_prob)
            itr_counter += 1
            
            if done:
                break

        print(f'On traning episode {i_episode}: Played {itr_counter}. Total episode reward {ep_reward}')
        # train the agent for the episode
        # Recalculate the total reward applying discounted factor
        discounts = [GAMMA ** i for i in range(len(rewards) + 1)]
        R = sum([a * b for a,b in zip(discounts, rewards)])

        policy_loss = []
        for log_prob in saved_log_probs:
            # Note that we are using Gradient Ascent, not Descent. 
            # So we need to calculate it with negative rewards.
            policy_loss.append(-log_prob * R)

        policy_loss = torch.cat(policy_loss).sum()
        
        # Backpropagation
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        scores_deque.append(sum(rewards))

        if np.mean(scores_deque) >= 195.0:
            print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            break

    return policy

def play(policy: Policy):

    print(f'Playing CartPole')
    version = 'v1'
    env_tag = f"CartPole-{version}"

    env = gym.make(id=env_tag, render_mode="human")

    state, _ = env.reset(seed=42)
    done = False
   
    step_counter = 0
    step_reward = 0.0
    while not done:

        action, log_prob = policy.act(state)

        observation, reward, done, truncated, info = env.step(action)

        step_reward += reward

        env.render()
        time.sleep(0.1)
        
        if done:
            break

        step_counter += 1
       

    print(f'Total reward {step_reward}')
    print(f'Played {step_counter}')


if __name__ == '__main__':
    policy = train()

    play(policy)
