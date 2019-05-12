import os
import gym
import time
import numpy
import pickle

#Load trained Q values
with open("MountainCarPolicy", "rb") as file:
    pi_star=pickle.load(file)

#Load state table
with open("MountainCarLookup", "rb") as file:
    dState=pickle.load(file)

#Setup environment
env = gym.make('MountainCar-v0')

#Navigate through environment based on policy
state = env.reset()
env.render()
state= state * numpy.array([10,100])
state = numpy.round(state,0).astype(int)
s = dState[str(state[0])+","+str(state[1])]
done = 0
while not done :
	#Perform the best action on the present state
	stateNext, r, done, _ = env.step(pi_star[s])
	stateNext= stateNext * numpy.array([10,100])
	stateNext = numpy.round(stateNext,0).astype(int)
	sNext = dState[str(stateNext[0])+","+str(stateNext[1])]
	env.render()
	s = sNext

env.close()

