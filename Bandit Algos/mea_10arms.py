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
def quickselect_median(l, pivot_fn=random.choice):
    if len(l) % 2 == 1:
        return quickselect(l, len(l) / 2, pivot_fn)
    else:
        return 0.5 * (quickselect(l, len(l) / 2 - 1, pivot_fn) +
                      quickselect(l, len(l) / 2, pivot_fn))


def quickselect(l, k, pivot_fn):
    """
    Select the kth element in l (0 based)
    :param l: List of numerics
    :param k: Index
    :param pivot_fn: Function to choose a pivot, defaults to random.choice
    :return: The kth element of l
    """
    if len(l) == 1:
        #assert k == 0
        return l[0]

    pivot = pivot_fn(l)

    lows = [el for el in l if el < pivot]
    highs = [el for el in l if el > pivot]
    pivots = [el for el in l if el == pivot]

    if k < len(lows):
        return quickselect(lows, k, pivot_fn)
    elif k < len(lows) + len(pivots):
        # We got lucky and guessed the median
        return pivots[0]
    else:
        return quickselect(highs, k - len(lows) - len(pivots), pivot_fn)
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
def ucb_run(rewards,nruns,cum_step_rew,opt_act,c):
    max_ind = np.argmax(rewards)
    step_rew = np.zeros(steps)
    cum_rew = np.zeros(narms)
    arm_sel = np.zeros(narms)
    for i in range(steps):
    
        if(i<narms):
            mind=i
        else:
            comp_ucb = np.zeros(narms)
            for j in range(narms):
                comp_ucb[j]=cum_rew[j]+c*(math.sqrt((2*math.log(i+1))/arm_sel[j]))
            mind = np.argmax(comp_ucb)
        if(max_ind==mind):
            opt_act[i]=opt_act[i]+1
        samp = np.random.normal(rewards[mind],1,1)
        cum_step_rew[i]=((cum_step_rew[i]*nruns)+samp)/(nruns+1)
        cum_rew[mind] = ((cum_rew[mind]*arm_sel[mind])+samp)/(arm_sel[mind]+1)
        arm_sel[mind]+=1

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

true_means =np.asarray(testbed)
opt_arm_index = np.argmax(true_means,1)
opt_arm_values = np.reshape(np.max(true_means,1),(numbandits,1))

#start = time.clock()
col=['r','g','k','b','y']
ed=[(0.1,0.1),(0.2,0.4),(0.8,0.5)]
fig1=plt.figure().add_subplot(111)#Average Learning curve for various eps - delta
fig2=plt.figure().add_subplot(111)#Number of Optimal arms present at each step
fig3=plt.figure().add_subplot(111)#
for p in range(3):
    eps = ed[p][0]
    delta = ed[p][1]
    #
    l = 1
    delta_it = delta / 2.0
    opt_arm_index1 = opt_arm_index
    rewards = []#algo measurement
    true_means1 = true_means
    eps_it = eps /4.0
    
    narms1 = int(narms)
    total_sam = 0
    opt_arm_cnts = []
    #iteration starts
    while narms1!=1:
        opt_arm_rem = 0
        sam = (np.log(3.0/delta_it)*4)/(eps_it**2)
        print(sam)
        cum_rew = np.zeros((numbandits , narms1))
        for i in range (int(sam)):
            i_rew = np.random.normal(true_means1,1) #best way to sample at once
            cum_rew = cum_rew + i_rew
            total_sam+=1
            rewards.append(np.mean(i_rew))
        cum_rew_avg = cum_rew/int(sam)  
        #rate_determining_step
        median_m2=[]
        for i in range(cum_rew_avg.shape[0]):
            median_m2.append(quickselect_median(cum_rew_avg[i]))
        medians = np.asarray(median_m2)
        print(medians)
        #print(median_m1)
        # print(medians.shape)
        
        #updates
        
        update_means = np.zeros((numbandits, narms1 - int(narms1/2)))
        
        for i in range(numbandits):
            k = 0
            for j in range(narms1):
                if(cum_rew_avg[i][j]>=medians[i]):
                    update_means[i][k]=true_means1[i][j]
                    k = k+1
                    if j == opt_arm_index1[i] :
                        opt_arm_rem += 1
        print(opt_arm_rem)             
        true_means1 = update_means
        opt_arm_index1 = np.argmax(true_means1,1)
        narms1 = narms1-int(narms1/2)
        opt_arm_cnts.append(opt_arm_rem)
        ####
        eps_it = eps_it*0.75
        delta_it = delta_it*0.5
        l=l+1
    sam = np.log(3.0/delta_it)*4/(eps_it**2)
        
    cum_rew = np.zeros((numbandits , narms1))
        
    for i in range (int(sam)):
        i_rew = np.random.normal(true_means1,1)
        cum_rew = cum_rew + i_rew
        total_sam+=1
        rewards.append(np.mean(i_rew))

    # print ('eps=',eps , '  delta= ',delta)
    # print (' ')
    # print(opt_arm_values.shape)
    # print(true_means1.shape)

    # absol = abs(opt_arm_values-true_means1)
    # print(absol)
    # opt_arms_correct = np.count_nonzero(absol)
    # #PAC optimality checking
    # #appromiate check
    # opt_arm_approx = np.count_nonzero(absol<eps) - opt_arms_correct
    # #number of bandits where its not true arm but appromiate",
    # print(opt_arm_approx)
    fig1.plot(range(total_sam),rewards,col[p])
    fig2.plot(range(1,l),opt_arm_cnts,col[p])




fig1.set_title('MEA Avg Reward for 10 arms')
fig1.set_ylabel('average reward')
fig1.set_xlabel('steps')
fig1.legend(("eps =  0.1, delta = 0.1","eps = 0.6 delta = 0.4","eps = 0.8 delta = 0.5"),loc='best')
fig2.set_title('MEA Optical action curve for 10 arms')
fig2.set_ylabel('Optimal action percentage')
fig2.set_xlabel('steps')
fig2.legend(("eps =  0.1, delta = 0.1","eps = 0.6 delta = 0.4","eps = 0.8 delta = 0.5"),loc='best')
plt.show()
exit()
print(total_sam)
steps =total_sam
T = range(1,steps+1)
fig3.plot(range(total_sam),rewards,'r')
fig3.plot(T,ucb_avg(2)[0],'g')
fig3.plot(T,soft_max_avg(0.1)[0],'k')
fig3.plot(T,eps_greedy_avg(0.1)[0],'b')
fig3.set_title('Comparision MEA vs ucb vs softmax vs eps-greedy')
fig3.set_ylabel('Avg reward')
fig3.set_xlabel('steps')
fig3.legend(('Mea eps = delta = ','ucb param = 5','softmax T =0.1','e-greedy e= 0.1'))
plt.show()

