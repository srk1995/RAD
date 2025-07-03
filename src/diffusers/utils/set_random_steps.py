import torch
import numpy as np

def set_random_steps(args):
    if "random_step" in args.exp_name:
        m_steps = np.random.choice(args.ddpm_num_steps, args.ddpm_mask_num_steps, replace=False)
        om_steps = np.delete(np.arange(args.ddpm_num_steps), m_steps)

    else:
        m_steps = np.arange(args.ddpm_mask_num_steps)
        om_steps = np.arange(args.ddpm_mask_num_steps, args.ddpm_num_steps)
        if 'geometric_mix_steps' in args.exp_name:
            num_mix_step = int(torch.distributions.geometric.Geometric(0.5).sample([1]).item())
            m = np.random.choice(np.arange(args.ddpm_mask_num_steps), num_mix_step, replace=False)
            om = np.random.choice(np.arange(args.ddpm_mask_num_steps, args.ddpm_num_steps), num_mix_step, replace=False)
            m_steps[m] = om
            om_steps[om - args.ddpm_mask_num_steps] = m

    m_steps.sort()
    om_steps.sort()

    random_steps = torch.tensor(np.hstack((m_steps, om_steps)))
    return random_steps, m_steps, om_steps
