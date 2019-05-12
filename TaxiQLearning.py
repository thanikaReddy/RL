import gym
import numpy
import pickle
import matplotlib.pyplot as plt

#Setup environment
#This environment has an observation space = Discrete(64) and action space = Discrete(4)
env = gym.make('Taxi-v2')

#Initialize Q for all state-action pairs, and Q Learning parameters
Q = numpy.random.random([env.observation_space.n,env.action_space.n])
alpha = 0.5
gamma = 0.99
kappa = 0.01
epsilon = 0.9
epsilonDecay = 0.995

#Number of episodes to train this on
epis = 20000

#Current episode number
epiCount = 1

#Average cumulative reward over all episodes
aveCumRew = 0

#Lists for plots
episodes = []
e = []
episodesCoarse = []
avgRews = []

while epiCount<=epis:
	#Initial state for a given episode
	s = env.reset()
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
		sNext, r, done, _ = env.step(a)

		#Update Q for this state, based on next state

		Q[s,a] = Q[s,a] + (alpha*((r+gamma*numpy.max(Q[sNext,:]))-Q[s,a]))
		s = sNext

		timeStep+=1

		#Update cumulative reward for this episode
		cumRew = r+(gamma*cumRew)

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
	if cumRew > aveCumRew:
		epsilon = epsilon * epsilonDecay

	#Every 1000 episodes, check if it is doing better
	if epiCount%1000 == 0:
		rew_avg = 0
		for i in range(100):
			obs = env.reset()
			done = 0
			cR = 0
			while (not done):
				a = numpy.argmax(Q[obs,:])
				obs, r, done, _ = env.step(a)
				cR = r + (gamma*cR)
			rew_avg += cR
		rew_avg = rew_avg/100
		print('Average reward per episode after {} episodes: {}'.format(epiCount,rew_avg))
		if rew_avg > 9.7:
			print('Taxi-v2 learned!')
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
for s in range(env.observation_space.n):
	pi_star[s] = numpy.argmax(Q[s,:])

#Dump policy to file
with open("TaxiPolicy", "wb") as file:
	pickle.dump(pi_star,file)


