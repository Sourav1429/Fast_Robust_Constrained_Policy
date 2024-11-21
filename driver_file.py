# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 12:47:16 2024

@author: gangu
"""

from itertools import product
import numpy as np
from Machine_Rep import gym_MR_env,Machine_Replacement,River_swim
import pickle

def form_probability_transition(P,policy,nS,nA):
    ret_mat = np.zeros((nS,nS))
    #print(policy.shape)
    for a in range(nA):
        for s in range(nS):
            for s_dash in range(nS):
                #print("(s,a)=",s,a,s_dash)
                #print(policy[int(s),int(a)])
                #print(P[int(a),int(s),int(s_dash)])
                ret_mat[int(s),int(s_dash)]+= policy[int(s),int(a)]*P[int(a),int(s),int(s_dash)]
    return ret_mat

def get_vf(P, R, pi, gamma=0.9):
    """
    Compute the value function for an MDP under a given policy using matrix operations.

    Parameters:
        P (np.ndarray): Transition probability matrix of shape (nS, nS, nA).
        R (np.ndarray): Reward matrix of shape (nS, nA).
        pi (np.ndarray): Policy matrix of shape (nS, nA).
        gamma (float): Discount factor (0 <= gamma < 1).

    Returns:
        np.ndarray: Value function vector of shape (nS,).
    """

    # Compute the policy transition matrix P_pi
    P_pi = P
    #print("pi=",pi)
    # Compute the policy reward vector R_pi
    R_pi = np.sum(pi * np.transpose(R), axis=1)
    #print(R_pi)

    # Solve the linear system (I - gamma * P_pi) V = R_pi
    I = np.eye(nS)
    V = np.linalg.solve(I - gamma * P_pi, R_pi)
    #print(V)
    return V

def find_min_vf_cf(policy,us,R,C,init,dist=0):
    nA,nS = R.shape
    v_list=[]
    c_list=[]
    policy = np.array(policy)
    for P in us:
        #print("P=",P)
        #print("policy=",policy)
        Tr = form_probability_transition(P, policy, nS, nA)
        #print("Transform_prob=",Tr)
        vf,cf = get_vf(Tr,R,policy),get_vf(Tr,C,policy)
        #print("VF=",vf)
        #print("CF=",cf)
        #break
        if(dist==0):
            vf,cf = vf[init],cf[init]
        else:
            vf,cf = np.dot(vf,init),np.dot(cf,init)
        v_list.append(vf)
        c_list.append(cf)
    return np.min(v_list),np.min(c_list)

def cross_product_of_keys(dictionary):
    """
    Finds the cross product of the elements of keys in a Python dictionary.

    Args:
        dictionary (dict): The dictionary to process.

    Returns:
        list: A list of tuples representing the cross product.
    """

    key_values = [dictionary[key] for key in dictionary]
    return list(product(*key_values))



if __name__=="__main__":
    env_type = 0
    env_choice=["Machine_Replacment","River_swim"]
    obj = None
    init_state = 0
    T = 10000
    alpha = 0.1
    lambda_ = 0.5
    b = 10
    eta = 0.1
    zi = 0.5
    us,pol_set = None,None
    if(env_type==0):
        obj = Machine_Replacement()
        init_state = 0
        with open("Uncertainity_set_Machine_Replacement_MDP","rb") as f:
            us = pickle.load(f)
        #print(us)
        f.close()

        with open("policies_MR","rb") as f:
            pol_set = pickle.load(f)
        #print(pol_set)
        f.close()
    else:
        obj = River_swim()
        init_state = 3
        with open("Uncertainity_set_River_Swim_MDP","rb") as f:
            us = pickle.load(f)
        #print(us)
        f.close()

        with open("policies_RS","rb") as f:
            pol_set = pickle.load(f)
        #print(pol_set)
        f.close()
    env = gym_MR_env(obj, init_state, T)
    nS,nA = env.observation_space_size(),env.action_space_size()
    R,C = obj.gen_expected_reward(),obj.gen_expected_cost()
    total_policies = cross_product_of_keys(pol_set)
    with open("all_policies_"+env_choice[env_type],"wb") as f:
        pickle.dump(total_policies,f)
    f.close()
    
    p0 = np.ones(len(total_policies))*1/len(total_policies)
    #vf_step,cf_step = [],[]
    for t in range(T):
        c_list = []
        for pol in range(len(total_policies)):
            vf,cf = find_min_vf_cf(total_policies[pol],us,R,C,init_state,dist=init_state)
            #vf_step.append(vf),cf_step.append(cf)
            #print(vf,cf)
            c_list.append(cf)
            p0[pol] = p0[pol]*np.exp(alpha*vf+lambda_*cf)
        #print(lambda_,eta*(b-np.dot(p0,np.array(c_list))))
        #print(np.max(lambda_+eta*(b-np.dot(p0,np.array(c_list))),0))
        #print(np.min(np.max(lambda_+eta*(b-np.dot(p0,np.array(c_list))),-10),zi))
        #break
        lambda_ = np.min([np.max([lambda_+eta*(b-np.dot(p0,np.array(c_list))),0]),zi])
        p0 = p0/np.sum(p0);
    print(p0)
    with open("Final_prob"+env_choice(env_type),"wb") as f:
        pickle.dump(p0,f)
    f.close()