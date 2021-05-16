import torch
import os

torch.manual_seed(42)


def one_hot_single_value(cur_val, total_vals, width):
    """Coverts cur_val into one-hot vector of size total_vals"""
    x = [0] * total_vals
    
    if len(cur_val) <= 1:
        idx = cur_val
    else:
        idx = cur_val[0] + width * cur_val[1]

    x[idx] = 1    
    return x


# def one_hot_two_value(cur_val_1, cur_val_2, total_vals_1, total_vals_2):
#     """Coverts cur_val_1, cur_val_2 into a vector of size total_vals_1 * total_vals_2"""
#     indices = torch.tensor(cur_val_2) * total_vals_1 + torch.argmax(torch.tensor(cur_val_1), dim=1)
#     x = torch.zeros((len(cur_val_2), total_vals_1 * total_vals_2))
#     for i, j in enumerate(indices):  # TODO: Vectorize
#         x[i, j] = 1
#     return x.squeeze()


def clip_tensor(input_tensor, min_val=-10., max_val=10.):
    """Clip learned reward and gamma"""
    return torch.clip(input=input_tensor, min=min_val, max=max_val)


def generate_trajectory(env, agent, policy, p_reward, p_gamma):
    """Generate trajectories"""
    states, actions = [], []
    inner_rewards, outer_rewards = [], []
    _ = env.reset()
    env.render()
    state, outer_reward, done = env.get_env_state()
    state_one_hot, action = agent.get_action(current_state=state, policy=policy)

    states.append(state_one_hot)
    actions.append(action)
    outer_rewards.append(outer_reward)

    while True:
        env.render()
        state, outer_reward, done, _ = env.step(action.item())
        state_one_hot, action = agent.get_action(current_state=state, policy=policy)

        states.append(state_one_hot)
        actions.append(action)
        outer_rewards.append(outer_reward)
        if done:
            break

    inner_rewards, gammas = agent.get_reward_gamma(states=states, actions=actions,
                                                   p_reward=p_reward, p_gamma=p_gamma)

    inner_rewards = clip_tensor(input_tensor=inner_rewards.squeeze())
    gammas = clip_tensor(input_tensor=gammas.squeeze(), min_val=0., max_val=0.9)

    return states, actions, inner_rewards, outer_rewards, gammas


def calc_grads(outputs, inputs, retain_graphs):
    """Calculates gradients of outputs w.r.t inputs.
        The order of outputs and inputs should be same."""
    out_grads = []

    for out, inp, retain_graph in zip(outputs, inputs, retain_graphs):
        grads = torch.autograd.grad(outputs=out,
                                    inputs=inp.parameters(),
                                    retain_graph=retain_graph)

        out_grads.append(torch.vstack([g.view(-1) for g in grads]).squeeze())

    return out_grads


def create_dir_and_save(model, parent_dir, filename):
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    torch.save(model.state_dict(), parent_dir + filename)


# class Basis(NeuralNet):
#     def __init__(self,  config):
#         super(Basis, self).__init__()
#
#         self.config = config
#
#         # Variables for normalizing state features
#         self.state_low = torch.tensor(config.env.observation_space.low, dtype=torch.float32, requires_grad=False,
#                                       device=config.device)
#         self.state_high = torch.tensor(config.env.observation_space.high, dtype=torch.float32, requires_grad=False,
#                                        device=config.device)
#         self.state_diff = self.state_high - self.state_low
#         self.state_dim = len(self.state_low)
#         self.flag = (self.state_diff > 1e3).any().item()  # Flag to Normalize or not
#
#         print("State Low: {} :: State High: {}".format(self.state_low, self.state_high))
#
#     def init(self):
#         print("State features: ", [(m, p.shape) for m, p in self.named_parameters()])
#         self.optim = self.config.optim(self.parameters(), lr=self.config.state_lr)
#
#     def preprocess(self, state):
#         if self.flag:
#             return state
#         else:
#             return (state - self.state_low)/self.state_diff
#
#
# class Fourier_Basis(Basis):
#     def __init__(self,  config):
#         super(Fourier_Basis, self).__init__(config)
#
#         dim = self.state_dim
#         order = self.config.fourier_order
#
#         if self.config.fourier_coupled:
#             if (order+1)**dim > 1000:
#                 raise ValueError("Reduce Fourier order please... ")
#
#             coeff = np.arange(0, order+1)
#             weights = torch.tensor(list(itertools.product(coeff, repeat=dim))).T  # size = n**d
#             self.get_basis = self.coupled
#             self.feature_dim = weights.shape[-1]
#         else:
#             weights = torch.arange(1, order + 1)
#             self.get_basis = self.uncoupled
#             self.feature_dim = weights.shape[-1] * dim
#
#         self.basis_weights = weights.type(torch.FloatTensor).requires_grad_(False).to(self.config.device)
#         self.dummy_param = torch.nn.Parameter(torch.rand(1).type(torch.FloatTensor))
#         self.init()