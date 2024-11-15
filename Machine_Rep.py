# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 18:00:18 2024

@author: gangu
"""
import numpy as np
class Machine_Replacement:
    def __init__(self,rep_cost=0.7,safety_cost=0.5,nS=4,nA=2):
        self.nS = nS;
        self.nA = nA;
        self.cost = np.linspace(0.1, 0.99,nS);
        self.rep_cost = rep_cost;
        self.safety_cost = safety_cost
    def gen_probability(self):
        self.P = np.zeros((self.nA,self.nS,self.nS));
        for i in range(self.nS):
            for j in range(self.nS):
                if(i<=j):
                    self.P[0,i,j]=(i+1)*(j+1);
                else:
                    continue;
            self.P[0,i,:]=self.P[0,i,:]/np.sum(self.P[0,i,:])
            self.P[1,i,0]=1;
        return self.P;
    def gen_reward(self):
        self.R=np.zeros((self.nA,self.nS,self.nS));
        for i in range(self.nS):
            self.R[0,i,:] = self.cost[i];
            self.R[1,i,0] = self.rep_cost+self.cost[0];
        return self.R;
    def gen_expected_reward(self):
        self.R = np.zeros((self.nA,self.nS));
        for i in range(self.nS):
            self.R[0,i] = self.cost[i];
            self.R[1,i] = self.rep_cost + self.cost[0];
        return self.R;
    def gen_expected_cost(self):
        self.C = np.zeros((self.nA,self.nS));
        for i in range(self.nS):
            self.C[0,i] = self.cost[i];
            self.C[1,i] = self.safety_cost + self.cost[0];
        return self.C;
    
class gym_MR_env:
    def __init__(self,mr_obj,init_state,T):
        self.P,self.R,self.C = mr_obj.gen_probability(),mr_obj.gen_expected_reward(),mr_obj.gen_expected_cost()
        self.init_state = init_state
        self.T = T
    
    def one_hot(self,s):
        one_hot_state = np.zeros(self.observation_space_size())
        one_hot_state[s] = 1
        return one_hot_state
        
    def reset(self):
        self.t=0
        self.state = self.one_hot(self.init_state)
        return self.init_state
    
    def step(self,action):
        rew = self.R[self.state,action]
        cost = self.C[self.state,action]
        next_state = np.random.choice(self.P[action,np.max(self.state),:])
        done = False
        if(self.t==self.T):
            done = True
        trunc = False
        return rew,self.one_hot(next_state),done,trunc,cost,None
    
    def observation_space_size(self):
        return len(self.P[0,0,:])
    
    def action_space_size(self):
        return self.P.shape[0]
