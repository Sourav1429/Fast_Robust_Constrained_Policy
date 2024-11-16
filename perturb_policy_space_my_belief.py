import numpy as np
from itertools import product
class perturb:
  def __init__(self,model,epsilon,n_actions,discretization_rate):
    self.model = model
    self.epsilon = epsilon
    self.n_actions = n_actions
    self.discretization_rate = discretization_rate
  def num_concentric_circles(self):
    return int(self.epsilon/self.discretization_rate)+1
  def get_all_policies(self,nS):
        pol_set = [[] for _ in range(self.num_policies)]
        for s in range(nS):
            k=0
            act = self.model.ret(s)
            pol_set[k].append(act)
            c = self.num_concentric_circles()
            for i in range(power(2,int(2*c))):
                
              
              
              
                  
              


nS,nA = 2,2
