import gym
import numpy
import pickle
import matplotlib.pyplot as plt

#Setup environment
env = gym.make('MountainCar-v0')

#Discretizing the state space and creating a lookup for Box(2, ) states to discrete numbers
#Number of states after discrtizing = 285
#Number of actions = 3
#Total number of state - action pairs = 855
dState = dict()
low = env.observation_space.low * numpy.array([10,100])
low = numpy.round(low,0).astype(int)
high = env.observation_space.high * numpy.array([10,100])
high = numpy.round(high,0).astype(int)
stateSize = 0;
for i in range(low[0], high[0]+1, 1):
	for j in range(low[1], high[1]+1, 1):
		dState[str(i)+","+str(j)] = stateSize
		stateSize+=1

#Number of episodes to train this on
epis = 20000

#Current episode number
epiCount = 1

#Average cumulative reward over all episodes
aveCumRew = 0

#Initialize Q for all state-action pairs, and Q Learning parameters
Q = numpy.random.uniform(low = -1, high = 1, 
                          size =(stateSize,env.action_space.n))
alpha = 0.2
gamma = 0.8
kappa = 0.01
epsilon = 1
epsilonDecay = (epsilon)/(epis)

#Lists for plots
episodes = []
e = []
episodesCoarse = []
avgRews = []
while epiCount<=epis:
	#Initial state for a given episode
	state = env.reset()
	state= state * numpy.array([10,100])
	state = numpy.round(state,0).astype(int)
	s = dState[str(state[0])+","+str(state[1])]
	timeStep = 1
	cumRew = 0

	while 1:
		#Determine whether to sample action randomly, or choose greedily
		
		randOrGreedy = numpy.random.uniform(0,1)

		if randOrGreedy<epsilon:
			#Randomly choose an action
			a = numpy.random.randint(env.action_space.n)
		else:
			
			#Greedily choose an action
			a = numpy.argmax(Q[s,:])

		#Perform the chosen action on the present state
		stateNext, r, done, _ = env.step(a)
		stateNext= stateNext * numpy.array([10,100])
		stateNext = numpy.round(stateNext,0).astype(int)
		sNext = dState[str(stateNext[0])+","+str(stateNext[1])]

		#Terminal reward
		#if done and stateNext[0]>=5:
		if done and stateNext[0]>=5:
			Q[s,a] = r
		else:
		#Update Q for this state, based on next state
			Q[s,a] = Q[s,a] + (alpha*((r+gamma*numpy.max(Q[sNext,:]))-Q[s,a]))

		s = sNext

		timeStep+=1

		#Update cumulative reward for this episode
		cumRew = r+(cumRew)

		#Check if episode has ended
		if done:
			epiCount+=1
			break

	#Add to average cumulative reward
	if epiCount == 1:
		aveCumRew = cumRew
	else:
		aveCumRew = kappa * cumRew + (1 - kappa)*aveCumRew



	#GLIE Exploration vs. Exploitation tuning - encourage exploitation
	#if cumRew > aveCumRew:
	if epsilon>0:
		epsilon = epsilon - epsilonDecay

	#Every 1000 episodes, check if it is doing better
	if epiCount%1000 == 0:
		rew_avg = 0
		for i in range(100):
			obs = env.reset()
			obs= obs * numpy.array([10,100])
			obs = numpy.round(obs,0).astype(int)
			dObs = dState[str(obs[0])+","+str(obs[1])]
			done = 0
			cR = 0
			while (not done):
				a = numpy.argmax(Q[dObs,:])
				obs, r, done, _ = env.step(a)
				obs= obs * numpy.array([10,100])
				obs = numpy.round(obs,0).astype(int)
				dObs = dState[str(obs[0])+","+str(obs[1])]
				cR = r + cR
			rew_avg += cR
		rew_avg = rew_avg/100
		print('Average reward per episode after {} episodes: {}'.format(epiCount,rew_avg))
		if rew_avg > -110:
			print('MountainCar-v0 learned!')
			break

		#For plots
		avgRews.append(rew_avg)
		episodesCoarse.append(epiCount)

	#For plots
	e.append(epsilon)
	episodes.append(epiCount)

fig, ax = plt.subplots(2, 1)
ax[0].plot(episodes, e, '.') 
ax[0].set_xlabel('Episode')
ax[0].set_ylabel('Epsilon (or exploration)')
ax[1].plot(episodesCoarse, avgRews, '--') 
ax[1].set_xlabel('Episode')
ax[1].set_ylabel('Average reward over 100 episodes')
plt.show()

print(Q)
#Find optimal policy from final Q values and store in dictionary
pi_star = dict()
for s in range(stateSize):
	pi_star[s] = numpy.argmax(Q[s,:])

#Dump policy to file
with open("MountainCarPolicy", "wb") as file:
	pickle.dump(pi_star,file)

#Dump state lookup to file
with open("MountainCarLookup", "wb") as file:
	pickle.dump(dState,file)


