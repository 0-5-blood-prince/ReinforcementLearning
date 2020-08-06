#!/usr/bin/env python

import click
import numpy as np
import gym
def include_bias(a):
    o = np.zeros(3)
    o[0]=1
    o[1]=a[0]
    o[2]=a[1]
    return o
def chakra_get_action(theta, ob, rng=np.random):
    ob_1 = include_bias(ob)
    mean = theta.dot(ob_1)
#     print("Mean is",mean)
#     return mean
    action = rng.normal(loc=mean, scale=1.)
    if(np.linalg.norm(action)>0.025):
        action= (action/np.linalg.norm(action))*0.025
    return action

@click.command()
@click.argument("env_id", type=str, default="chakra")
def main(env_id):
    # Register the environment
    rng = np.random.RandomState(42)

    if env_id == 'chakra':
        import rlpa2
        env = gym.make('chakra-v0')
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
    print(np.shape(theta))
    N = 100
    while N>=0:
        ob = env.reset()
        print(ob)
#         ob = np.zeros(2)
        done = False
        # Only render the first trajectory
        # Collect a new trajectory
        rewards = []
        while not done:
            action = get_action(theta, ob, rng=rng)
#             print("Action is",a)
            next_ob, rew, done, _ = env.step(action)
           
#             print("next ob is",next_ob)
            ob = next_ob
            if done:
                print("----------------------------")
#             env.render("human")
            rewards.append(rew)

        print("Episode reward: %.2f                                               Done" % np.sum(rewards))
        N-=1

if __name__ == "__main__":
    main()
