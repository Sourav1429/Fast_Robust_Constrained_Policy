# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 17:32:28 2024

@author: gangu
"""

import numpy as np
import torch

class perturb_nominal_:
    def __init__(self,accessed_vector,epsilon,is_function = 1):
        self.accessed_vector = accessed_vector
        self.eps = epsilon
        self.is_function = is_function
    def fit(self,d=1):
        if self.is_function == 1:
            return self.perturb_function()
        else:
            return self.perturb_vector(d)
    def perturb_vector(self,set_size = 1):
        vect_set = []
        for _ in range(set_size):
            dir_ = np.random.choice([-1,1],size = len(self.accessed_vector))
            perturb_val_vect = self.eps*np.random.random(len(self.accessed_vector))*dir_
            vect_set.append(self.accessed_vector + perturb_val_vect)
        return vect_set
            
class perturb:
  def __init__(self,model,epsilon,n_actions,discretization_rate):
    self.model = model
    self.epsilon = epsilon
    self.n_actions = n_actions
    self.discretization_rate = discretization_rate
  def num_policies(self):
    return np.power(2,self.n_actions)*self.num_concentric_circles()
  def num_concentric_circles(self):
    return int(self.epsilon/self.discretization_rate)
  def perturb_policy_space(self):
    high = self.num_concentric_circles()
    radii_perturb = np.random.uniform(0,high)+1
    amount_perturb = np.array([np.random.choice([-1,1])*radii_perturb for _ in range(self.n_actions)])
    return amount_perturb
  def perturb_policy(self,state):
    return self.model(state,deterministic = False) + torch.tensor(self.perturb_policy_space());       

        