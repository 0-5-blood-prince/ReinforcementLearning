from gym import Env
# from gym.envs.registration import register
from gym.utils import seeding
from gym import spaces
import numpy as np
import random
# class GridWorld(Env):
# metadata = {
#         'render.modes': ['human', 'rgb_array'],
#         'video.frames_per_second': 50
#     }
class gridenv1(Env):
    def __init__(self,n=12,m=12):
        self.grid = np.zeros((m,n))
        self.m = m
        self.n = n
        self.n_actions = 4
        self.stateSpace = [i for i in range(self.m*self.n)]
        self.startstates = [60,72,120,132]
        self.startind = random.randint(0,3)
        self.agentposition = self.startstates[self.startind]
        ##Terminal state info say A
        self.stateSpace.remove(3*(self.n-1))#A index
        self.stateSpaceplus = [i for i in range(self.m*self.n)]
        self.actionSpace = {'U':-self.n,'D':self.n,'L':-1,'R':1}
        self.possibleActions = ['U','D','L','R']
        #Rewards
        self.rewards  = np.zeros((m,n))
        
        for i in self.stateSpace:
            x = i // self.m
            y = i%self.n
            self.rewards[x][y]=0
            if y in range(3,9):
                if y == 8:
                    if x in range(2,7):
                        self.rewards[x][y]-=1
                else:
                    if x in range(2,9):
                        self.rewards[x][y]-=1
            if y in range(4,8):
                if y == 7:
                    if x in range(3,6):
                        self.rewards[x][y]-=1
                else:
                    if x in range(3,8):
                        self.rewards[x][y]-=1
            if y in range(5,7):
                if y == 6:
                    if x==4:
                        self.rewards[x][y]-=1
                else:
                    if x in range(4,7):
                        self.rewards[x][y]-=1
        self.rewards[2][n-3]+=10  #A reward
        self.setState(self.agentposition)
                    
                    
        
    def isTerminal(self,state):
        return state in self.stateSpaceplus and state not in self.stateSpace
    def getAgentRnC(self):
        x = self.agentposition // self.m
        y = self.agentposition % self.n
        return x,y
    def setState(self,state):
        x,y = self.getAgentRnC()
#         print(x)
#         print(y)
        self.grid[x][y]=-1
        self.agentposition = state
        x,y = self.getAgentRnC()
        self.grid[x][y]=1
    def offgrid(self,newState,oldState):
        if newState not in self.stateSpaceplus:
#             print("off")
            return True
        elif oldState % self.m == 0 and newState % self.m == self.m - 1:
            return True
        elif oldState % self.m == self.m-1 and newState % self.m == 0:
            return True
        else:
            return False
    def step(self,action):
        action = self.possibleActions[action]
        x,y = self.getAgentRnC()
        
        prob_w = random.uniform(0,1)
        wind = False
        if(prob_w<=0.5):
            wind = True
        #stoc = False
        prob_s = random.uniform(0,1)
        action_m = action
        if(prob_s<=0.9):
            action_m = action
        else:
            t = random.randint(1,3)
            c =1
            for a in self.actionSpace:
                if(a==action):
                    continue
                else:
                    if(c==t):
                        action_m = a
                        break
                    c+=1
                
        resultingstate = self.agentposition + self.actionSpace[action_m]
        trans_state = 0
        reward = 0
        #here Reward will be after windapplication Think 
        #Say you got into terminal by moving west and wind blows you east you
        #remain in the same position
        if not self.offgrid(resultingstate,self.agentposition):
#             self.setState(resultingstate)
            trans_state = resultingstate
        else:
#             print("is off",resultingstate)
            trans_state = self.agentposition
#             print(trans_state)
        
        if(wind):
            result_wind = trans_state + self.actionSpace['R']
            if not self.offgrid(result_wind,trans_state):
#                 self.setState(result_wind)
                trans_state = result_wind
            else:
#                 print("is off",result_wind)
                trans_state = self.agentposition
#                 print(trans_state)
        self.setState(trans_state)    
        x_end = trans_state // self.m
        y_end = trans_state % self.n
        isT = self.isTerminal(self.agentposition)
#         print("state")
#         print(trans_state)
#         print(x_end)
#         print(y_end)
    
        return trans_state,self.rewards[x_end][y_end],isT,None
        
    def reset(self):
        self.grid = np.zeros((self.m,self.n))
        self.startind = random.randint(0,3)
        self.agentposition = self.startstates[self.startind]
        return self.agentposition
    def render(self):
        print('-----------------')
        for row in self.grid:
            for col in row:
                if col==0:
                    print('-',end =" ")
                elif col==1 or col==-1:
                    print('X',end =" ")
            print('\n')
        print('------------------')
