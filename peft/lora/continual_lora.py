import torch
import torch.nn as nn
import math
import numpy as np

class ContinualLora(nn.Module):
    def __init__(self, depth, dim, rank, lora_alpha=1, lora_qkv=[False, False, True], lora_out=False, lora_fc1=False, lora_fc2=False):
        super().__init__()
        self.depth = depth
        if lora_qkv is None:
            lora_qkv = [True, True, True]
        self.r = rank
        self.dim = dim
        self.lora_alpha = lora_alpha
        self.scaling = torch.tensor(self.lora_alpha / self.r)
        self.qkv_lora = lora_qkv
        self.q_lora, self.k_lora, self.v_lora = self.qkv_lora[0], self.qkv_lora[1], self.qkv_lora[2]
        self.out_lora = lora_out
        self.fc1_lora = lora_fc1
        self.fc2_lora = lora_fc2
        assert isinstance(lora_qkv, list) and len(self.qkv_lora) == 3 
        self.create_parameters()
        self.reset_parameters()

    def create_parameters(self):
        attributes = ['q_lora', 'k_lora', 'v_lora', 'out_lora', 'fc1_lora', 'fc2_lora']
        for attr_name in attributes:
            cond =  getattr(self, attr_name)
            if attr_name in ['q_lora', 'k_lora', 'v_lora', 'out_lora']:
                if cond:
                    setattr(self, attr_name+'_A', nn.Parameter(torch.zeros((self.depth, self.dim, self.r))))
                    setattr(self, attr_name+'_B', nn.Parameter(torch.zeros((self.depth, self.r, self.dim))))
                else:
                    setattr(self, attr_name+'_A', torch.zeros((self.depth, self.dim, self.r)))
                    setattr(self, attr_name+'_B', torch.zeros((self.depth, self.r, self.dim)))
            elif attr_name == 'fc1_lora':
                if cond:
                    setattr(self, attr_name+'_A', nn.Parameter(torch.zeros((self.depth, self.dim, self.r))))
                    setattr(self, attr_name+'_B', nn.Parameter(torch.zeros((self.depth, self.r, self.dim * 4))))
                else:
                    setattr(self, attr_name+'_A', torch.zeros((self.depth, self.dim, self.r)))
                    setattr(self, attr_name+'_B', torch.zeros((self.depth, self.r, self.dim * 4)))
            elif attr_name == 'fc2_lora':
                if cond:
                    setattr(self, attr_name+'_A', nn.Parameter(torch.zeros((self.depth, self.dim * 4, self.r))))
                    setattr(self, attr_name+'_B', nn.Parameter(torch.zeros((self.depth, self.r, self.dim))))
                else:
                    setattr(self, attr_name+'_A', torch.zeros((self.depth, self.dim * 4, self.r)))
                    setattr(self, attr_name+'_B', torch.zeros((self.depth, self.r, self.dim)))
            else:
                raise NotImplementedError

    def reset_parameters(self):
        params = ['q_lora_A', 'q_lora_B', 'k_lora_A', 'k_lora_B', 'v_lora_A', 'v_lora_B', 'out_lora_A', 'out_lora_B', 'fc1_lora_A', 'fc1_lora_B', 'fc2_lora_A', 'fc2_lora_B']
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
        params = ['q_lora_A', 'q_lora_B', 'k_lora_A', 'k_lora_B', 'v_lora_A', 'v_lora_B', 'out_lora_A', 'out_lora_B', 'fc1_lora_A', 'fc1_lora_B', 'fc2_lora_A', 'fc2_lora_B']
        for param_name in params:
            if not isinstance(getattr(self, param_name), nn.Parameter):
                setattr(self, param_name, getattr(self, param_name).to(device))

    def cal_delta_w(self, depth, x=None, device=None):
        q_lora = torch.mm(self.q_lora_A[depth], self.q_lora_B[depth])
        k_lora = torch.mm(self.k_lora_A[depth], self.k_lora_B[depth])
        v_lora = torch.mm(self.v_lora_A[depth], self.v_lora_B[depth])
        
        # TODO: .to(device) is a time-cost operation
        if x is not None:
            self.delta_qkv = torch.cat([q_lora.to(x.device), k_lora.to(x.device), v_lora.to(x.device)], dim=-1) * self.scaling
            self.delta_out = torch.mm(self.out_lora_A[depth].to(x.device), self.out_lora_B[depth].to(x.device)) * self.scaling
            self.delta_fc1 = torch.mm(self.fc1_lora_A[depth].to(x.device), self.fc1_lora_B[depth].to(x.device)) * self.scaling
            self.delta_fc2 = torch.mm(self.fc2_lora_A[depth].to(x.device), self.fc2_lora_B[depth].to(x.device)) * self.scaling
        if device is not None:
            self.delta_qkv = torch.cat([q_lora.to(device), k_lora.to(device), v_lora.to(device)], dim=-1) * self.scaling
            self.delta_out = torch.mm(self.out_lora_A[depth].to(device), self.out_lora_B[depth].to(device)) * self.scaling
            self.delta_fc1 = torch.mm(self.fc1_lora_A[depth].to(device), self.fc1_lora_B[depth].to(device)) * self.scaling
            self.delta_fc2 = torch.mm(self.fc2_lora_A[depth].to(device), self.fc2_lora_B[depth].to(device)) * self.scaling

        return self.delta_qkv, self.delta_out, self.delta_fc1, self.delta_fc2

    def forward(self, x, task_id=-1, depth_id=-1, train=False, model_num=-1, position='qkv'):
        out = dict()
        self.to_device(x.device)
        assert position in ('qkv', 'out', 'fc1', 'fc2')
        if position == 'qkv':
            q = self.q_lora_A[depth_id] @ self.q_lora_B[depth_id]
            k = self.k_lora_A[depth_id] @ self.k_lora_B[depth_id]
            v = self.v_lora_A[depth_id] @ self.v_lora_B[depth_id]
            w = torch.cat([q.to(x.device), k.to(x.device), v.to(x.device)], dim=-1) * self.scaling
        elif position == 'out':
            w = self.out_lora_A[depth_id] @ self.out_lora_B[depth_id] * self.scaling
        elif position == 'fc1':
            w = self.fc1_lora_A[depth_id] @ self.fc1_lora_B[depth_id] * self.scaling
        else:    
            w = self.fc2_lora_A[depth_id] @ self.fc2_lora_B[depth_id] * self.scaling

        out['lora_value'] = torch.einsum('bld,dz->blz', x, w)
        return out
    
    def after_task(self, *args):
        self.reset_parameters()
        