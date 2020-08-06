import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import math
import random
import csv
import copy
import os
import glob
#number of arms
narms = 10

#test bed 
numbandits = 2000
mu = 0 
var = 1
testbed = []
for i in range(numbandits):
    s = np.random.normal(mu,var,narms)
    testbed.append(s)
steps = 1000
#testbed is a list an element being a bandit with rewards of arms
def greedy_run(rewards,eps,nruns,cum_step_rew,opt_act):
	#one run consider one bandit 
	#cum_step_rew is the cumultative step reward calculated till now update it instead of storing
    max_ind = np.argmax(rewards)
    step_rew = np.zeros(steps)
    cum_rew = np.zeros(narms)
    arm_sel = np.zeros(narms)
    for i in range(steps):
        toss = random.uniform(0,1)
        if(toss >= eps):
        	#Exploitation
            mind = np.argmax(cum_rew)
        else:
        	#Exploration
            mind = random.randint(0,9)
        if(max_ind==mind):
            opt_act[i]=opt_act[i]+1
        samp = np.random.normal(rewards[mind],1,1)
        cum_step_rew[i]=((cum_step_rew[i]*nruns)+samp)/(nruns+1)
        cum_rew[mind] = ((cum_rew[mind]*arm_sel[mind])+samp)/(arm_sel[mind]+1)
        arm_sel[mind]+=1

def eps_greedy_avg(eps):
    cum_step_rew = np.zeros(steps)
    opt_act = np.zeros(steps)
#     print(cum_step_rew)
    for i in range(numbandits):
#         print(i)
        greedy_run(testbed[i],eps,i,cum_step_rew,opt_act) 
    for i in range(steps) :
    	opt_act[i] = (opt_act[i]/numbandits)*100 
    #cum_step_rew consists of average Learning process for all bandits
    #opt_act consists of optimal selection of arms same as above
    return cum_step_rew,opt_act
col=['r','g','k','b','y']
#plt.rc('text',usetex=True)
fig1=plt.figure().add_subplot(111)
fig2=plt.figure().add_subplot(111)
#fig1 - Optimal action percentage
#fig2 -Average Reward Curve
T = range(1,steps+1)
fig1.plot(T,eps_greedy_avg(0)[1],col[0])
fig1.plot(T,eps_greedy_avg(0.01)[1],col[1])
fig1.plot(T,eps_greedy_avg(0.1)[1],col[2])
fig1.plot(T,eps_greedy_avg(0.5)[1],col[3])
fig1.plot(T,eps_greedy_avg(1)[1],col[4])
fig1.set_title('epsilon-greedy:Optimal action curve for 10 arms')
fig1.set_ylabel('Optimal action Percentage')
fig1.set_xlabel('Plays')
fig1.legend(("e = 0","e = 0.01","e = 0.1","e = 0.5","e = 1"),loc='best')
T = range(1,steps+1)
fig2.plot(T,eps_greedy_avg(0)[0],col[0])
fig2.plot(T,eps_greedy_avg(0.01)[0],col[1])
fig2.plot(T,eps_greedy_avg(0.1)[0],col[2])
fig2.plot(T,eps_greedy_avg(0.5)[0],col[3])
fig2.plot(T,eps_greedy_avg(1)[0],col[4])
fig2.set_title('epsilon-greedy:Avg Reward Learning curve for 10 arms')
fig2.set_ylabel('Average Reward')
fig2.set_xlabel('Plays')
fig2.legend(("e = 0","e = 0.01","e = 0.1","e = 0.5","e = 1"),loc='best')
plt.show()