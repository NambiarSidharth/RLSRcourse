# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 21:05:30 2019

@author: Surendran Nambiar
"""

import gym
import numpy as np
import random
import time
import pickle
env=gym.make('Taxi-v2')
gamma=0.8
epsilon=0.2
lr=0.001
Q=np.zeros((env.observation_space.n,env.action_space.n))

def generate_random_action(env):
    return np.random.choice(env.action_space.n)
def run_game(env,episodes=150000):
    for i in range(episodes):
        curr_state=env.reset()
        print(i)
        while True:
            if random.uniform(0,1)<epsilon:
                #explore
                #print("exploring")
                action=generate_random_action(env)
                new_state,reward,done,info=env.step(action)
                #env.render()
                #time.sleep(0.1)
                #updating q table
                #print(Q[curr_state,action],new_state,reward,done)
                Q[curr_state,action]=Q[curr_state,action]+(lr*(reward+(gamma*np.max(Q[new_state,:]))-Q[curr_state,action]))
                #print(Q[curr_state,action])

                curr_state=new_state
                if done or reward==-10:
                    break
            else:
                #exploit
                #print("exploiting")
                action=np.argmax(Q[curr_state,:])
                new_state,reward,done,info=env.step(action)
                #updating q table
                #env.render()
                #time.sleep(0.1)
                #print(Q[curr_state,action],new_state,reward,done)
                Q[curr_state,action]=Q[curr_state,action]+(lr*(reward+(gamma*np.max(Q[new_state,:]))-Q[curr_state,action]))
                #print(Q[curr_state,action])                
                curr_state=new_state
                if done or reward==-10:
                    break

run_game(env)
env.render()

def play_one_turn(env):
    curr_state=env.reset()
    env.render()
    time.sleep(1)
    while True:
        action=np.argmax(Q[curr_state,:])
        new_state,reward,done,info=env.step(action)
        print(reward,done)
        env.render()
        time.sleep(1)
        curr_state=new_state
        if done or reward==-10:
                    break

play_one_turn(env)

with open("Taxi_v2_qtable.pkl", 'wb') as f:
    pickle.dump(Q, f)