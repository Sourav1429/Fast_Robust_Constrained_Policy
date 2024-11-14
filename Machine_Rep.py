# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 18:00:18 2024

@author: gangu
"""
import numpy as np
class Machine_Replacement:
    def __init__(self,rep_cost=0.7,safety_cost=0.8,nS=4,nA=2):
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