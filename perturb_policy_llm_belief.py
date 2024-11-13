#Implementation 2 and correct according to chatgpt but I think wrong
class perturb:
  def __init__(self,model,epsilon,n_actions,discretization_rate):
    self.model = model
    self.epsilon = epsilon
    self.n_actions = n_actions
    self.discretization_rate = discretization_rate
  def num_policies(self):
    return self.num_concentric_circles()
  def num_concentric_circles(self):
    return int(self.epsilon/self.discretization_rate)
  def perturb_policy_space(self):
    high = self.num_concentric_circles()
    radii_perturb = np.random.uniform(0,high)+1
    choice = np.random.choice([-1,1])
    amount_perturb = np.array([choice*radii_perturb for _ in range(self.n_actions)])
    return amount_perturb
  def perturb_policy(self,state):
    return self.model(state,deterministic = False) + torch.tensor(self.perturb_policy_space());
