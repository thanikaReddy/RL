import os
import gym
import time
import numpy
import pickle

#Load trained Q values
with open("FrozenLakePolicy8x8", "rb") as file:
    pi_star=pickle.load(file)

#Setup environment
env = gym.make('FrozenLake8x8-v0')

#Navigate through environment based on policy
s = env.reset()
env.render()
done = 0
while not done :
	#Perform the best action on the present state
	sNext, r, done, _ = env.step(pi_star[s])
	env.render()
	time.sleep(0.1)
	s = sNext

