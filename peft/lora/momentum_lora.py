import torch
import torch.nn as nn
import math
import numpy as np
import copy

class MomentumLora(nn.Module):
    def __init__(self, depth, dim, rank, lora_alpha=1, momentum=0.99):
        super().__init__()
        self.depth = depth
        self.r = rank
        self.dim = dim
        self.lora_alpha = lora_alpha
        self.scaling = torch.tensor(self.lora_alpha / self.r)
        self.momentum = momentum
        self.create_parameters()
        self.reset_parameters()

    def create_parameters(self):
        attributes = ['k_lora', 'v_lora']
        for attr_name in attributes:
            setattr(self, attr_name+'_A', nn.Parameter(torch.zeros((self.depth, self.dim, self.r))))
            setattr(self, attr_name+'_B', nn.Parameter(torch.zeros((self.depth, self.r, self.dim))))
                
        self.q_lora_A = torch.zeros((self.depth, self.dim, self.r))
        self.q_lora_B = torch.zeros((self.depth, self.r, self.dim))
            
    def reset_parameters(self):
        params = ['k_lora_A', 'k_lora_B', 'v_lora_A', 'v_lora_B']
        for param_name in params:
            param = getattr(self, param_name)
            if isinstance(param, nn.Parameter):
                if param_name.endswith('_A'):
                    d, _, _ = param.shape
                    for i in range(d):
                        nn.init.kaiming_uniform_(param[i], a=math.sqrt(5))
                else:
                    nn.init.zeros_(param)

    def to_device(self, device):
        params = ['q_lora_A', 'q_lora_B', 'k_lora_A', 'k_lora_B', 'v_lora_A', 'v_lora_B']
        for param_name in params:
            if not isinstance(getattr(self, param_name), nn.Parameter):
                setattr(self, param_name, getattr(self, param_name).to(device))
        for param_name in params:
            if hasattr(self, param_name + '_mom'):
                setattr(self, param_name + '_mom', getattr(self, param_name + '_mom').to(device))

    def cal_delta_w(self, depth, x=None, device=None):
        q_lora = torch.mm(self.q_lora_A[depth], self.q_lora_B[depth])
        k_lora = torch.mm(self.k_lora_A[depth], self.k_lora_B[depth])
        v_lora = torch.mm(self.v_lora_A[depth], self.v_lora_B[depth])
        
        # TODO: .to(device) is a time-cost operation
        if x is not None:
            self.delta_qkv = torch.cat([q_lora.to(x.device), k_lora.to(x.device), v_lora.to(x.device)], dim=-1) * self.scaling
        if device is not None:
            self.delta_qkv = torch.cat([q_lora.to(device), k_lora.to(device), v_lora.to(device)], dim=-1) * self.scaling
            
        return self.delta_qkv

    def forward(self, x, task_id=-1, depth_id=-1, train=False, old=False):
        out = dict()
        self.to_device(x.device)
        if not old or task_id == 0:
            q = self.q_lora_A[depth_id] @ self.q_lora_B[depth_id]
            k = self.k_lora_A[depth_id] @ self.k_lora_B[depth_id]
            v = self.v_lora_A[depth_id] @ self.v_lora_B[depth_id]
            w = torch.cat([q.to(x.device), k.to(x.device), v.to(x.device)], dim=-1) * self.scaling
            out['lora_value'] = torch.einsum('bld,dz->blz', x, w)
        if old and task_id > 0:
            q = self.q_lora_A_mom[depth_id] @ self.q_lora_B_mom[depth_id]
            k = self.k_lora_A_mom[depth_id] @ self.k_lora_B_mom[depth_id]
            v = self.v_lora_A_mom[depth_id] @ self.v_lora_B_mom[depth_id]
            w = torch.cat([q.to(x.device), k.to(x.device), v.to(x.device)], dim=-1) * self.scaling
           
            out['lora_value'] = torch.einsum('bld,dz->blz', x, w)

        return out
    
    def copy_parameters(self, task_id=-1, device=None):
        params = ['k_lora_A', 'k_lora_B', 'v_lora_A', 'v_lora_B']
        if task_id == 0:          
            for param_name in params:
                if isinstance(getattr(self, param_name), nn.Parameter):
                    setattr(self, param_name + '_mom', copy.deepcopy(getattr(self, param_name).detach().clone()))
                else:
                    setattr(self, param_name + '_mom', copy.deepcopy(getattr(self, param_name)))
                setattr(self, param_name + '_mom', getattr(self, param_name + '_mom').to(device))
        else:
            for param_name in params:
                if isinstance(getattr(self, param_name), nn.Parameter):
                    setattr(self, param_name + '_mom', self.momentum * getattr(self, param_name+'_mom') + (1 - self.momentum) * copy.deepcopy(getattr(self, param_name).detach().clone()))
                else:
                    setattr(self, param_name + '_mom', copy.deepcopy(getattr(self, param_name)))
                setattr(self, param_name + '_mom', getattr(self, param_name + '_mom').to(device))
    
    def after_task(self, task_id=-1, device=None):
        self.copy_parameters(task_id=task_id, device=device)
        