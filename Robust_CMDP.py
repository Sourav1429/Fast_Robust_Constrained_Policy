# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 11:44:23 2024

@author: gangu
"""
import torch
import numpy as np
from perturb_policy_space_my_belief import perturb
from Machine_Rep import Machine_Replacement,gym_MR_env
from perturbations import perturb_nominal_
import pickle

def create_prob_space(P,nS,nA):
    for s in range(nS):
        for a in range(nA):
            P[a,s,: ] = P[a,s,:]/np.sum(P[a,s])
    return P

def build_uncertainity_set(P_dict,nS,nA,sets):
    uncertain_set = []
    for s_a_set in range(sets):
        P = np.zeros((nA,nS,nS))
        for s in range(nS):
            for a in range(nA):
                sel_P = P_dict[(s,a)][s_a_set]
                sel_P = sel_P/np.sum(sel_P)
                P[a,s] = sel_P
        uncertain_set.append(P)
    return uncertain_set
                

mr_obj = Machine_Replacement()
env = gym_MR_env(mr_obj,0,1000)
lambda_ = 0.9
lr = 0.01
b = 8
psi = 0.5
nS,nA = mr_obj.nS,mr_obj.nA
P0 = np.ones((env.action_space_size(),env.observation_space_size(),env.observation_space_size()))
P0 = create_prob_space(P0, nS, nA)
print(P0)
policy_function = torch.load('qnetwork_dqn_model_Machine_Replacement')
policy_perturbation_epsilon = 0.1
policy_discretization_rate = 0.01
perturbed_policy = perturb(policy_function,policy_perturbation_epsilon,env.action_space_size(),policy_discretization_rate)
n_policies = perturbed_policy.num_policies()
p = np.ones(n_policies)/n_policies
T = 10000
un_eps = 0.1
P_dict = {}
no_of_perturb_Set = 5
for s in range(env.observation_space_size()):
  for a in range(env.action_space_size()):
    pr_nom = perturb_nominal_(P0[a,s],un_eps,0)
    P_dict[(s,a)] = pr_nom.perturb_vector(no_of_perturb_Set)
#print(P_dict)
uncertain_set = build_uncertainity_set(P_dict, nS, nA, no_of_perturb_Set)
with open("Uncertainity_set_Machine_Replacement_MDP","wb") as f:
    pickle.dump(uncertain_set, f)

#for t in range(2):
    
