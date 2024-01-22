import torch
import torch.nn as nn
import math
import numpy as np

class HideLoraPool(nn.Module):
    def __init__(self, pool_size, depth, dim, rank, lora_alpha=1):
        super().__init__()
        self.r = rank
        self.lora_alpha = lora_alpha
        self.scaling = torch.tensor(self.lora_alpha / self.r)
        self.depth = depth
        self.pool_size = pool_size
        self.dim = dim

        self.create_parameters()
        self.reset_parameters()

    def create_parameters(self):
        attributes = ['k_lora', 'v_lora']
        for attr_name in attributes:
                setattr(self, attr_name+'_A', nn.Parameter(torch.zeros((self.pool_size, self.depth, self.dim, self.r))))
                setattr(self, attr_name+'_B', nn.Parameter(torch.zeros((self.pool_size, self.depth, self.r, self.dim))))
                
        self.q_lora_A = torch.zeros((self.pool_size, self.depth, self.dim, self.r))
        self.q_lora_B = torch.zeros((self.pool_size, self.depth, self.r, self.dim))

    def reset_parameters(self):
        params = ['k_lora_A', 'k_lora_B', 'v_lora_A', 'v_lora_B']
        for param_name in params:
            param = getattr(self, param_name)
            if isinstance(param, nn.Parameter):
                if param_name.endswith('_A'):
                    p, d, _, _ = param.shape
                    for i in range(p):
                        for j in range(d):
                            nn.init.kaiming_uniform_(param[i][j], a=math.sqrt(5))
                else:
                    nn.init.zeros_(param)
        
    def to_device(self, device):
        params = ['q_lora_A', 'q_lora_B', 'k_lora_A', 'k_lora_B', 'v_lora_A', 'v_lora_B']
        for param_name in params:
            if not isinstance(getattr(self, param_name), nn.Parameter):
                setattr(self, param_name, getattr(self, param_name).to(device))

        
    def cal_delta_w(self, x=None, device=None, task_id=-1, depth=-1):
        k_lora = torch.mm(self.k_lora_A[task_id, depth], self.k_lora_B[task_id, depth])
        v_lora = torch.mm(self.v_lora_A[task_id, depth], self.v_lora_B[task_id, depth])
        q_lora = torch.zeros((self.dim, self.dim))
        # TODO: .to(device) is a time-cost operation
        if x is not None:
            self.delta_w = torch.cat([q_lora.to(x.device), k_lora.to(x.device), v_lora.to(x.device)], dim=-1) * self.scaling
        if device is not None:
            self.delta_w = torch.cat([q_lora.to(device), k_lora.to(device), v_lora.to(device)], dim=-1) * self.scaling
        return self.delta_w
    
    def forward(self, x, task_id=-1, depth_id=-1, train=False, **kwargs):
        out = dict()
        self.to_device(x.device)
        if train:
            assert isinstance(task_id, int)
            q = self.q_lora_A[task_id][depth_id] @ self.q_lora_B[task_id][depth_id]
            k = self.k_lora_A[task_id][depth_id] @ self.k_lora_B[task_id][depth_id]
            v = self.v_lora_A[task_id][depth_id] @ self.v_lora_B[task_id][depth_id]
            w = torch.cat([q.to(x.device), k.to(x.device), v.to(x.device)], dim=-1) * self.scaling
            out['lora_value'] = torch.einsum('bld,dz->blz', x, w)
            return out
            
        else:
            assert isinstance(task_id, list) or isinstance(task_id, torch.Tensor)
            q = torch.bmm(self.q_lora_A[task_id, depth_id], self.q_lora_B[task_id, depth_id])
            k = torch.bmm(self.k_lora_A[task_id, depth_id], self.k_lora_B[task_id, depth_id])
            v = torch.bmm(self.v_lora_A[task_id, depth_id], self.v_lora_B[task_id, depth_id])
            w = torch.cat([q.to(x.device), k.to(x.device), v.to(x.device)], dim=-1) * self.scaling
            out['lora_value'] = torch.bmm(x, w) # B x L x 3dim
        return out

    def after_task(self, task_id, device=None):
        pass
