import numpy as np
from itertools import product
from Machine_Rep import Machine_Replacement,gym_MR_env
import torch
import pickle

class get_policy_combinations:
    def __init__(self,model,states,nS,nA,d,epsilon):
        self.model = model
        self.states = states
        self.nS = nS
        self.nA = nA
        self.d = d
        self.k = int(epsilon/d)
    def __ret_policies__(self):
        return generate_epsilon_close_distributions(self.model, self.states, self.nS,self.nA, self.d, self.k)

def one_hot(nS,s):
    ret_val = np.zeros(nS)
    ret_val[s] = 1
    return ret_val

def make_probs(pr):
    pr = np.exp(pr)
    return pr/np.sum(pr)

def create_prob_space(P,nS,nA):
    for s in range(nS):
        for a in range(nA):
            P[a,s,: ] = P[a,s,:]/np.sum(P[a,s])
    return P
        

def generate_epsilon_close_distributions(model, states, nS,nA, d, k):
    """
    Generate all epsilon-close distributions for a given model and discretization.

    Parameters:
        model (function): A function that takes a state `s` and returns a probability distribution over actions.
        states (list): A list of states for which to generate distributions.
        nA (int): Number of actions in the action space.
        d (float): Discretization step size.
        k (int): Scaling factor for epsilon (epsilon = k * d).

    Returns:
        dict: A dictionary where keys are states and values are lists of epsilon-close distributions for each state.
    """
    epsilon_close_distributions = {}

    for s in states:
        # Get the original probability distribution for state s
        s1 = torch.tensor(one_hot(nS, s)).float()
        original_probs = model(s1).cpu().detach().numpy()
        original_probs = make_probs(original_probs)
        #print(original_probs)
        
        # Generate perturbed ranges for each action
        action_ranges = [
            np.clip(
                np.arange(min(p, p - k*d), max(p, p + k * d) + d, d),
                p-k*d, p+k*d
            )
            for p in original_probs
        ]
        #print("============")
        #print(action_ranges)
        
        # Generate all combinations of perturbed probabilities
        all_combinations = list(product(*action_ranges))
        
        # Filter combinations to ensure they sum to 1 (within a tolerance)
        #valid_distributions = make_distribution(all_combination)
        valid_distributions = [
            np.array(comb)
            for comb in all_combinations
            if np.isclose(sum(comb), 1, atol=1e-6)
        ]
        #print("********************")
        #print(valid_distributions)
        
        epsilon_close_distributions[s] = valid_distributions

    return epsilon_close_distributions
# Define a model that gives a probability distribution
mr_obj = Machine_Replacement()
P,R,C = mr_obj.gen_probability(),mr_obj.gen_expected_reward(),mr_obj.gen_expected_cost()
init_state,T = 0,1000
env = gym_MR_env(mr_obj, init_state, T)
lambda_ = 0.9
lr = 0.01
b = 8
psi = 0.5
nS,nA = mr_obj.nS,mr_obj.nA
P0 = np.ones((env.action_space_size(),env.observation_space_size(),env.observation_space_size()))
P0 = create_prob_space(P0, nS, nA)

policy_function = torch.load('qnetwork_dqn_model_Machine_Replacement')
policy_perturbation_epsilon = 0.1
policy_discretization_rate = 0.05

gp = get_policy_combinations(policy_function, np.arange(nS,dtype=np.int16),nS, nA, policy_discretization_rate, policy_perturbation_epsilon)
possible_policies = gp.__ret_policies__()

print(possible_policies)
with open("policies_MR","wb") as f:
    pickle.dump(possible_policies,f)
print("=+=+=+=+=+===+++++++++++++++=================+=+=+")


'''def model(s):
    if s == "s0":
        return np.array([0.6, 0.4])
    elif s == "s1":
        return np.array([0.3, 0.7])
    return np.array([0.5, 0.5])

# Define parameters
states = ["s0", "s1"]  # Example state set
nA = 2  # Number of actions
d = 0.01  # Step size for discretization
k = 3  # Scaling factor for epsilon (epsilon = k * d)

# Generate epsilon-close distributions
epsilon_distributions = generate_epsilon_close_distributions(model, states, nA, d, k)

# Display results
for state, distributions in epsilon_distributions.items():
    print(f"State: {state}, Number of Distributions: {len(distributions)}")
    print("Example Distributions:")
    for dist in distributions:  # Show only the first 5 distributions
        print(dist)
    print()'''
