class perturb:
  def __init__(self,model,epsilon,n_actions):
    self.model = model
    self.epsilon = epsilon
    self.n_actions = n_actions
  def perturb_policy_space(self):
    low = np.ones((n_actions,1))*-self.epsilon
    high = np.ones((n_actions,1))*self.epsilon
    amount_perturb = np.random.uniform(low,high,size(n_actions,1))
    return amount_perturb
  def perturb_policy(self,state):
    return self.model(state,deterministic = False) + torch.tensor(self.perturb_policy_space());
