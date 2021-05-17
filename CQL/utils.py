import numpy as np
import torch
import math
from torch.distributions import Distribution, Normal
import d4rl
import gym


def choice(x):
    if x: return 0
    else: return 1
    
class DicToDataset(torch.utils.data.Dataset):
    def __init__(self,dictionary):
        self.dict = dictionary
        self.keys = dictionary.keys()
        
    def __getitem__(self, index):
        
        obs = self.dict['observations'][index]
        act = self.dict['actions'][index]
        nxt_obs = self.dict['next_observations'][index]
        rwd = self.dict['rewards'] [index]
        tmls =self.dict['terminals'][index]
        
        return {'observations': obs, 'actions':act, 'next_observations': nxt_obs, 'rewards': rwd, 'terminals':tmls}
    
    def __len__(self):
        return self.dict['observations'].shape[0]
    

    
def load_replaymemory_dataloader(env,batch_size=256):
    dic= d4rl.qlearning_dataset(env)
    terminals = list(map(choice,dic['terminals']))
    dic['terminals'] = terminals 
    dataset = DicToDataset(dic)
    #convert terminals to 0/1 array
    
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=2,shuffle=True)
    return loader



def soft_update(target, source, tau):
	for target_param, param in zip(target.parameters(), source.parameters()):
		target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
	for target_param, param in zip(target.parameters(), source.parameters()):
		target_param.data.copy_(param.data)






#--- code from CQL paper / rlkit for lo_prob sampling for behavior cloninng


def atanh(x):
    one_plus_x = (1 + x).clamp(min=1e-6)
    one_minus_x = (1 - x).clamp(min=1e-6)
    return 0.5*torch.log(one_plus_x/ one_minus_x)




class TanhNormal(Distribution):
    """
    Represent distribution of X where
        X ~ tanh(Z)
        Z ~ N(mean, std)

    Note: this is not very numerically stable.
    """
    def __init__(self, normal_mean, normal_std, epsilon=1e-6):
        """
        :param normal_mean: Mean of the normal distribution
        :param normal_std: Std of the normal distribution
        :param epsilon: Numerical stability epsilon when computing log-prob.
        """
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.normal = Normal(normal_mean, normal_std)
        self.epsilon = epsilon

    def sample_n(self, n, return_pre_tanh_value=False):
        z = self.normal.sample_n(n)
        if return_pre_tanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def log_prob(self, value, pre_tanh_value=None):
        """

        :param value: some value, x
        :param pre_tanh_value: arctanh(x)
        :return:
        """
        if pre_tanh_value is None:
            pre_tanh_value = torch.log(
                (1+value) / (1-value)
            ) / 2
        return self.normal.log_prob(pre_tanh_value) - torch.log(
            1 - value * value + self.epsilon
        )

    def sample(self, return_pretanh_value=False):
        """
        Gradients will and should *not* pass through this operation.

        See https://github.com/pytorch/pytorch/issues/4620 for discussion.
        """
        z = self.normal.sample().detach()

        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    
