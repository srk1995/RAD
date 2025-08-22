# Copyright (C) 2024 Sora Kim
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import torch
import numpy as np

def set_random_steps(args):
    m_steps = np.arange(args.ddpm_mask_num_steps)
    om_steps = np.arange(args.ddpm_mask_num_steps, args.ddpm_num_steps)

    m_steps.sort()
    om_steps.sort()

    random_steps = torch.tensor(np.hstack((m_steps, om_steps)))
    return random_steps, m_steps, om_steps
