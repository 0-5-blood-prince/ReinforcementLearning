#!/usr/bin/env python

import click
import numpy as np
import gym
from sklearn.linear_model import LinearRegression
def include_bias(a):
    o = np.zeros(3)
    o[0]=1
    o[1]=a[0]
    o[2]=a[1]
    return o
def chakra_get_action(theta, ob, rng=np.random):
	#This get action function is same for chakra and vishamC
    ob_1 = include_bias(ob)
    mean = theta.dot(ob_1)
    action = rng.normal(loc=mean, scale=1.)
    if(np.linalg.norm(action)>0.025):
        action= (action/np.linalg.norm(action))*0.025
    return action
def traject_episode(env,theta,rng,get_action):
	#Returns Trajectories of Specified env and theta
    done = False
    ob = env.reset()
    rewards = []
    ep_list = []
    steps=0
    while not done:
        Trans = []
        Trans.append(ob)
#         print("State is",ob)
        action = get_action(theta, ob, rng=rng)
#         print("Action is",action)
        Trans.append(action)
        next_ob, rew, done, r = env.step(action)
        a = np.zeros(2)
        a[0]=rew
        a[1]=0
        Trans.append(a)
        ep_list.append(Trans)
#         print("next ob is",a)
        ob = next_ob
        steps+=1
        rewards.append(rew)
#         env.render("human")
        if done:
            break
#             print("----------------------------")
        
    return ep_list,rewards
def deriv_log(action,state,theta):
	#returns the log derivative 
	#Log function was asked but it was useless 
    mean = theta.dot(include_bias(state))
    L = np.zeros((2,3))
    m1 = -((action[0]-mean[0]))
    L[0][0]=-(m1*1)
    L[0][1]=-(m1*state[0])
    L[0][2]=-(m1*state[1])
    m2 = -((action[1]-mean[1]))
    L[1][0]=-(m2*1)
    L[1][1]=-(m2*state[0])
    L[1][2]=-(m2*state[1])
    return L
@click.command()
@click.argument("env_id", type=str, default="chakra")
def main(env_id):
    # Register the environment
    rng = np.random.RandomState(42)
    gamma = 0.6
    alpha = 0.05
    alp_decay = 1.0
    if env_id == 'chakra':
        import rlpa2
        env = gym.make('chakra-v0')
        env._max_episode_steps =40
        get_action = chakra_get_action
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
    elif env_id == 'vishamC':
        import rlpa2
        env = gym.make('vishamC-v0')
        env._max_episode_steps =40
        get_action = chakra_get_action
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
    else:
        raise ValueError(
            "Unsupported environment ")

    env.seed(42)

    # Initialize parameters
    theta = rng.normal(scale=0.01, size=(action_dim, obs_dim + 1))
    MAX_ITR = 20
    itr = 0
    al=[]
    while itr<=MAX_ITR:        
        
        BATCH_SIZE =50
        n_samples = 0
        Grads = []
        Avg_reward = 0
        Avg_length = 0

        #Below Commented Snippet was for Baseline

#         from sklearn.preprocessing import PolynomialFeatures
#         from sklearn import linear_model
#         poly = PolynomialFeatures(degree=1)
#         clf=linear_model.LinearRegression()
#         new_trajec ,rewards= traject_episode(env,theta,rng,get_action)
# #         print(rewards)
#         L = len(new_trajec)
#         t = L-1
#         badvants = np.zeros(L)
#         R = 0
#         bX_t = []
#         while t>=0:
#             R = gamma*R + new_trajec[t][2][0]
#             badvants[t]=R
#             bX_t.append(new_trajec[t][0])
#             t-=1
        
        
#         bx = poly.fit_transform(bX_t)
#         clf.fit(bx,badvants)

        while n_samples<BATCH_SIZE:
            new_trajec ,rewards= traject_episode(env,theta,rng,get_action)
            
            grad = np.zeros((2,3))
            L = len(new_trajec)
#             print("Reward sum is ",np.sum(rewards),"Length is ",L)
            Avg_reward = (Avg_reward*n_samples + (np.sum(rewards)))/(n_samples+1)
            Avg_length = (Avg_length*n_samples+L)/(n_samples+1)
            t = L-1
            derivs = []
            advants = np.zeros(L)
            R = 0
            X_t =[]
            while t>=0:
                R = gamma*R + new_trajec[t][2][0]
                advants[t]=R
                X_t.append(new_trajec[t][0])
                t-=1

            # if itr>=0:
            # 	X = poly.fit_transform(X_t)
            # 	baseline = clf.predict(X)
            # else:
            #     baseline=np.zeros(len(X_t))
            #Calulate the gradient contribution for this Trajectory
            for i in range(L):
                deriv=deriv_log(new_trajec[i][1],new_trajec[i][0],theta)
                grad = grad + deriv * (advants[i])

#             reg.fit(np.asarray(X_t),np.asarray(advants))
	
            norm_grad = grad /(np.linalg.norm(grad)+1e-8)

            Grads.append(norm_grad)
            n_samples+=1
        Gradient = np.zeros((2,3))
        for i in range(BATCH_SIZE):
            Gradient+=Grads[i]
        Gradient/=BATCH_SIZE
        print("Gradient is ",Gradient)
        print("theta is ",theta)
        
        theta = theta+ (alpha * Gradient)
        print("itr is _________________________________ ",itr)
        print("Average reward is _______________________",Avg_reward)
        print("Average length is _______________________",Avg_length)
        al.append(Avg_length)
        alpha *=alp_decay
        
        itr+=1

if __name__ == "__main__":
    main()
