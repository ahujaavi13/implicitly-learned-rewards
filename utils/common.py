import torch
import os

torch.manual_seed(42)


def one_hot_single_value(cur_val, total_vals):
    """Coverts cur_val into one-hot vector of size total_vals"""
    x = [0] * total_vals
    x[cur_val] = 1
    return x


def one_hot_two_value(cur_val_1, cur_val_2, total_vals_1, total_vals_2):
    """Coverts cur_val_1, cur_val_2 into a vector of size total_vals_1 * total_vals_2"""
    indices = torch.tensor(cur_val_2) * total_vals_1 + torch.argmax(torch.tensor(cur_val_1), dim=1)
    x = torch.zeros((len(cur_val_2), total_vals_1 * total_vals_2))
    for i, j in enumerate(indices):  # TODO: Vectorize
        x[i, j] = 1
    return x.squeeze()


def clip_tensor(input_tensor, min_val=-10., max_val=10.):
    """Clip learned reward and gamma"""
    return torch.clip(input=input_tensor, min=min_val, max=max_val)


def generate_trajectory(env, agent, policy, p_reward, p_gamma, max_trajectory_len):
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

    for t in range(max_trajectory_len):
        env.render()
        state, outer_reward, done = env.step(action.item())
        state_one_hot, action = agent.get_action(current_state=state, policy=policy)

        states.append(state_one_hot)
        actions.append(action)
        outer_rewards.append(outer_reward)

        if done:
            break

    inner_rewards, gammas = agent.get_reward_gamma(states=states, actions=actions,
                                                   p_reward=p_reward, p_gamma=p_gamma)

    inner_rewards = clip_tensor(input_tensor=inner_rewards.squeeze())
    gammas = clip_tensor(input_tensor=gammas.squeeze(), min_val=0., max_val=1.)

    return states, actions, inner_rewards, outer_rewards, gammas


def calc_grads(outputs, inputs, retain_graphs):
    """Calculates gradients of outputs w.r.t inputs.
        The order of outputs and inputs should be same."""
    out_grads = []

    for out, inp, retain_graph in zip(outputs, inputs, retain_graphs):
        grads = torch.autograd.grad(outputs=out,
                                    inputs=[p for p in inp.parameters()],
                                    retain_graph=retain_graph)
        out_grads.append(torch.vstack([g.view(-1) for g in grads]).squeeze())

    return out_grads


def create_dir_and_save(model, parent_dir, filename):
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    torch.save(model.state_dict(), parent_dir + filename)
