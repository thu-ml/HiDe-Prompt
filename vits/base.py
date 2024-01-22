import torch.nn as nn

class MlpHead(nn.Module):
    def __init__(self, input_dim=768, ratio=2, output_dim=10, drop_out=0.0) -> None:
        super().__init__()
        hidden_dim = input_dim * ratio
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(drop_out)
    def forward(self, x):
        x = self.fc2(self.drop(self.layer_norm(self.activation(self.fc1(x)))))
        return x
    def forward_features(self, x):
        return self.layer_norm(self.activation(self.fc1(x)))
    def forward_head(self, x):
        return self.fc2(x)
        

