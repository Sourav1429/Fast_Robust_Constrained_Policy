# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 17:49:11 2024

@author: gangu
"""
import numpy as np


class Machine_Replacement:
    def __init__(self,rep_cost=0.7,nS=4,nA=2):
        self.nS = nS;
        self.nA = nA;
        self.cost = np.linspace(0.1, 0.99,nS);
        self.rep_cost = rep_cost;
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



class min_value_func:
    def __init__(self,P_set,n_actions,actions,init_state,gamma):
        self.P_set = P_set
        self.n_actions = n_actions
        self.actions = actions
        self.init_state = init_state
        self.gamma = gamma
    def find_vf(self,preparedP,R,policy):
        I = np.eye(self.n_actions)
        print(I)
        Q = self.gamma*np.dot(policy[self.init_state],np.transpose(preparedP))
        print(Q)
        return np.dot(np.linalg.pinv(I-Q),np.dot(policy[self.init_state,:],R))
    def obtain_robust_vf(self,policy,R):
        state = self.init_state
        vf = []
        for P in self.P_set:
            preparedP = []
            for a in self.actions:
                preparedP.append(P[a,state,:])
            preparedP = np.array(preparedP)
            V = self.find_vf(preparedP,R,policy)
            vf.append(V)
        pos = np.argmin(vf)
        return (vf[pos],pos)

mr_obj = Machine_Replacement();
P,R = mr_obj.gen_probability(),mr_obj.gen_expected_reward()
print(P)
print(R)
state = 0
actions = [0,1]
preparedP = []
for a in actions:
    preparedP.append(P[a,state,:])
preparedP = np.array(preparedP)
preparedR = np.array([R[state,a] for a in actions])
print(preparedP,preparedR)


policy = np.array([[1,0],
                   [1,0],
                   [0,1],
                   [0,1]])
Q = 0.95*np.dot(np.transpose(policy[0]),np.transpose(preparedP))
print(Q)

model = min_value_func(P, len(actions), actions, 0, 0.95)
print(model.find_vf(preparedP,preparedR,policy))
            
        