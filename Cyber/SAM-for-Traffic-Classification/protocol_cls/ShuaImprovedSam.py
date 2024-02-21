import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch import Tensor
#Notes: 
# 1) The paper uses only 1 head for computations, would recomend using more heads for better performance-- better gpus' are available and cheaper now
# 2) original does a very simple addition of the byte and position vectors for positional encoding. Shoudl add in more complex positional encoding
#    they don't mention the method in the paper though. I used the positional encoding from the original transformer paper,
#    with the implementation from the pytorch tutorial

class PositionalEncoding(nn.Module): #https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model).unsqueeze(0) # modify the positional encoding tensor to be [1, max_len, d_model]
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:, :x.size(1)] 
        return self.dropout(x)
    
class Cnn(nn.Module):
    def __init__(self, in_channels, kernel_amount, kernel_size,dropout=0.1):
        super(Cnn, self).__init__()
        self.cnn = nn.Conv1d(in_channels=in_channels, out_channels=kernel_amount, kernel_size=kernel_size)
        self.pooling = nn.AdaptiveMaxPool1d(1) #AdaptiveMaxPool1d is used to make the model more flexible to different input
                                               # sizes for different packet lengths
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x):
        conv_out = self.cnn(x)
        conv_out = F.relu(conv_out)
        pooled = self.pooling(conv_out).squeeze(-1)  # Remove the last dimension after pooling, to fit the linear layer
        return self.dropout(pooled)
    
class SAM(nn.Module):
    def __init__(self, num_class,max_byte_len, embed_dim=256, num_heads=4, #Increased number of heads
                 conv1_kernel_size=3,
                 conv2_kernel_size=4,
                 kernels=256,
                 attn_dropout=0.1):
        super(SAM, self).__init__()
        self.bytes_embedding = nn.Embedding(num_embeddings=300, embedding_dim=embed_dim)  # Assuming 300 as a generic vocabulary size for demonstration
        self.position_embedding = PositionalEncoding(embed_dim, max_len=max_byte_len)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=attn_dropout) #paper uses 1 head for computational reasons
        self.conv1 = Cnn(in_channels=embed_dim, kernel_amount=kernels, kernel_size=conv1_kernel_size)
        self.conv2 = Cnn(in_channels=embed_dim, kernel_amount=kernels, kernel_size=conv2_kernel_size)
        input_features = embed_dim*2 # Concatenating the output of the two CNNs
        self.fc = nn.Linear(input_features, num_class)
        
    def forward(self, x, pos):
        # pos_embedding = self.position_embedding(pos)  # pos: [seq_len, batch_size, embed_dim]
        # byte_embedding = self.bytes_embedding(x)  # x: [seq_len, batch_size, embed_dim]
        # x = byte_embedding + pos_embedding
        x = self.position_embedding(self.bytes_embedding(x))
        x = x.permute(1, 0, 2)  # Adjust x to [batch_size, seq_len, embed_dim] for MultiheadAttention
        attn_output, weights = self.multihead_attn(x, x, x) 
        attn_output = attn_output.permute(1,2,0) 
        out1 = self.conv1(attn_output)
        out2 = self.conv2(attn_output)
        combined_cnn_output = torch.cat((out1, out2), dim=1)  # Concatenate along the feature dimension
        out = self.fc(combined_cnn_output)
        if not self.training: #taken from original implementation, to work with the rest of the training script
            return F.softmax(out, dim=-1).max(1)[1], torch.mean(weights, dim=-2)
        return out