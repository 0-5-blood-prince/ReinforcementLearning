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
def greedy_run(rewards,eps,nruns,cum_step_rew,opt_act):
    max_ind = np.argmax(rewards)
    step_rew = np.zeros(steps)
    cum_rew = np.zeros(narms)
    arm_sel = np.zeros(narms)
    for i in range(steps):
        toss = random.uniform(0,1)
        if(toss >= eps):
            mind = np.argmax(cum_rew)
        else:
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
    return cum_step_rew,opt_act



def soft_max_run(rewards,nruns,cum_step_rew,opt_act,beta):
    #one run consider one bandit 
    #cum_step_rew is the cumultative step reward calculated till now update it instead of storing
    max_ind = np.argmax(rewards)
    step_rew = np.zeros(steps)
    cum_rew = np.zeros(narms)
    arm_sel = np.zeros(narms)
    prob = np.zeros(narms)
    for i in range(steps):
        toss = random.uniform(0,1)
        tot = 0
        for j in range(narms):
            prob[j] = pow(3.14,cum_rew[j]/beta)
            tot +=prob[j]
        for j in range(narms):
            prob[j] = (prob[j])/tot
        #finding the probability with weightage for every arm based on estimated mean
        cur = 0
        for j in range(narms):
            cur+=prob[j]
            if(toss <= cur):
                #finding the arm according to probabilities
                mind = j
                break
        if(max_ind==mind):
            opt_act[i]=opt_act[i]+1
        samp = np.random.normal(rewards[mind],1,1)
        cum_step_rew[i]=((cum_step_rew[i]*nruns)+samp)/(nruns+1)
        cum_rew[mind] = ((cum_rew[mind]*arm_sel[mind])+samp)/(arm_sel[mind]+1)
        arm_sel[mind]+=1

def soft_max_avg(beta):
    cum_step_rew = np.zeros(steps)
    opt_act = np.zeros(steps)
#     print(cum_step_rew)
    for i in range(numbandits):
#         print(i)
        soft_max_run(testbed[i],i,cum_step_rew,opt_act,beta) 
    for i in range(steps) :
        opt_act[i] = (opt_act[i]/numbandits)*100 
    return cum_step_rew,opt_act


col=['r','g','k','b','y']
#plt.rc('text',usetex=True)
fig1=plt.figure().add_subplot(111)
fig2=plt.figure().add_subplot(111)

T = range(1,steps+1)
fig1.plot(T,soft_max_avg(0.02)[1],col[0])
fig1.plot(T,soft_max_avg(0.1)[1],col[1])
fig1.plot(T,soft_max_avg(1)[1],col[2])
fig1.plot(T,soft_max_avg(10)[1],col[3])
fig1.set_title('Softmax:Optimal action curve vs Plays for 10 arms')
fig1.set_ylabel('Optimal action Percentage')
fig1.set_xlabel('Plays')
fig1.legend(("T = 0.02","T = 0.1","T = 1","T = 10"),loc='best')
T = range(1,steps+1)
fig2.plot(T,soft_max_avg(0.02)[0],col[0])
fig2.plot(T,soft_max_avg(0.1)[0],col[1])
fig2.plot(T,soft_max_avg(1)[0],col[2])
fig2.plot(T,soft_max_avg(10)[0],col[3])
fig2.set_title('Softmax:Avg Reward Learning curve vs Plays for 10 arms')
fig2.set_ylabel('Average Reward')
fig2.set_xlabel('Plays')
fig2.legend(("T = 0.02","T = 0.1","T = 1","T = 10"),loc='best')
plt.show()