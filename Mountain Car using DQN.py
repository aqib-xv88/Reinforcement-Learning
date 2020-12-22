#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gym
import random


# In[2]:


env = gym.make("MountainCar-v0") # may be needs memory
states = env.observation_space.shape[0]
actions = env.action_space.n


# In[4]:


episodes = 2
for episodes in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0
    
    while not done:
        env.render()
        action = random.choice([0,2]) # what's possible action
        n_state, reward, done, info = env.step(action)
        score+=reward
    print('Episode:{} Score:{}'.format(episodes, score))


# In[5]:


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam


# In[13]:


def build_model(states, actions):
    model = Sequential()
    model.add(Flatten(input_shape=(1,states)))
    model.add(Dense(24, activation='relu')) # change the model
    model.add(Dense(24, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model


# In[14]:


model = build_model(states, actions)


# In[1]:


print(model.summary())


# In[46]:


from rl.agents import DQNAgent
from rl.policy import EpsGreedyQPolicy 
from rl.memory import SequentialMemory


# In[49]:


def build_agent(model, actions):
    policy = EpsGreedyQPolicy() # change policy for different problems
    memory = SequentialMemory(limit=500000, window_length=1) 
    dqn = DQNAgent(model=model, memory=memory, policy=policy, 
                  nb_actions=actions, nb_steps_warmup=10, 
                  target_model_update=1e-2,batch_size=64) 
    
    return dqn


# In[2]:


dqn = build_agent(model, actions)
dqn.compile(Adam(lr=1e-3), metrics=['mae']) 
dqn.fit(env, nb_steps=100000, visualize=True, verbose=1)


# In[ ]:


scores = dqn.test(env, nb_episodes=10, visualize=True)
print(np.mean(scores.history['episode_reward']))


# In[3]:


_ = dqn.test(env, nb_episodes=10, visualize=True)

