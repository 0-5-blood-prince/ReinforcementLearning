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
        cur = 0
        for j in range(narms):
            cur+=prob[j]
            if(toss <= cur):
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
def ucb_run(rewards,nruns,cum_step_rew,opt_act,C):
    max_ind = np.argmax(rewards)
    step_rew = np.zeros(steps)
    cum_rew = np.zeros(narms)
    arm_sel = np.zeros(narms)
    avg =0
    for i in range(narms):
        c=np.random.normal(rewards[i],1,1)
        arm_sel[i]+=1
        avg+=c
    avg/=narms
    for i in range(narms):
        cum_rew[i]=0
    cum_step_rew[0]=(cum_step_rew[0]*nruns +avg )/(nruns+1)
    for i in range(1,steps-1):
            comp_ucb = np.zeros(narms)
            for j in range(narms):
                comp_ucb[j]=cum_rew[j]+C*( math.sqrt(  ( 2* (math.log(i+1)) )/arm_sel[j])  )
            mind = np.argmax(comp_ucb)
            if(max_ind==mind):
                opt_act[i]=opt_act[i]+1
            samp = np.random.normal(rewards[mind],1,1)
            cum_step_rew[i]=((cum_step_rew[i]*nruns)+samp)/(nruns+1)
            cum_rew[mind] = ((cum_rew[mind]*arm_sel[mind])+samp)/(arm_sel[mind]+1)
            arm_sel[mind]+=1.

def ucb_avg(c):
    cum_step_rew = np.zeros(steps)
    opt_act = np.zeros(steps)
#     print(cum_step_rew)
    for i in range(numbandits):
#         print(i)
        ucb_run(testbed[i],i,cum_step_rew,opt_act,c)
    for i in range(steps) :
        opt_act[i] = (opt_act[i]/numbandits)*100 
    return cum_step_rew,opt_act
col=['r','g','k','b','y']
#plt.rc('text',usetex=True)
fig1=plt.figure().add_subplot(111)
fig2=plt.figure().add_subplot(111)
fig3=plt.figure().add_subplot(111)
T = range(1,steps+1)
#fig1.plot(T,ucb_avg(0.01)[1],col[0])
A = ucb_avg(0.1)
B = ucb_avg(2)
fig1.plot(T,A[1],col[1])

fig1.plot(T,B[1],col[2])
# fig1.plot(T,ucb_avg(10)[1],col[3])
fig1.set_title('UCB1:Optimal action curve for 10 arms')
fig1.set_ylabel('Optimal action Percentage')
fig1.set_xlabel('Plays')
fig1.legend(("C = 0.1","C = 2"),loc='best')
T = range(1,steps+1)
#fig2.plot(T,ucb_avg(0.01)[0],col[0])
fig2.plot(T,A[0],col[1])

fig2.plot(T,B[0],col[2])
# fig2.plot(T,ucb_avg(10)[0],col[3])
fig2.set_title('UCB1:Avg Reward Learning curve for 10 arms')
fig2.set_ylabel('Average Reward')
fig2.set_xlabel('Plays')
fig2.legend(("C = 0.1","C = 2"),loc='best')
# plt.show()


# best epsilon greedy e= 0.1
# best eps



eps = 0.1
temp = 0.1
ucb_p = 2

fig3.plot(T,eps_greedy_avg(eps)[0],col[0])
fig3.plot(T,soft_max_avg(temp)[0],col[1])
fig3.plot(T,ucb_avg(ucb_p)[0],col[2])
fig3.set_title('Comparision between E-greedy # Softmax # UCB1')
fig3.set_ylabel('Average Reward')
fig3.set_xlabel('Steps')
fig3.legend(("e-greedy e = 0.1","softmax Temp = 0.1","UCB1 param = 2"),loc='best')
plt.show()