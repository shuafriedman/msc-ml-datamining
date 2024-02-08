import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#Notes: 
# 1) original paper only uses one head because of complexity
# 2) original does a very simple addition of the byte and position vectors for positional encoding. Shoudl add in more complex positional encoding
#    they don't mention the method in the paper though
# 3) In the paper, they write that they use 2 1-d cnns combined, this doesn't exist in the code though
# 4) The paper uses only 1 head for computations, would recomend using more heads for better performance-- better gpus' are available and cheaper now

class Cnn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Cnn, self).__init__()
        self.cnn = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.pooling = nn.AdaptiveMaxPool1d(1)
    def forward(self, x):
        conv_out = self.cnn(x)
        conv_out = F.relu(conv_out)
        pooled = self.pooling(conv_out).squeeze(-1)  # Remove the last dimension after pooling
        return pooled
    
class SAM(nn.Module):
    def __init__(self, num_classes, max_len, embed_dim=128, num_heads=1, kernel_size=3, filters=128):
        super(SAM, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=300, embedding_dim=embed_dim)  # Assuming 300 as a generic vocabulary size for demonstration
        
        # Multi-head Attention
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads) #paper uses 1 head for computational reasons
        self.conv1 = Cnn(in_channels=embed_dim, out_channels=filters, kernel_size=kernel_size)
        self.conv2 = Cnn(in_channels=embed_dim, out_channels=filters, kernel_size=kernel_size)
        self.pool = nn.AdaptiveMaxPool1d(1)
        # Fully Connected Layer for Classification
        self.fc = nn.Linear(256, num_classes)
        
    def forward(self, x, pos):
        x = self.embedding(x)  # x: [seq_len, batch_size, embed_dim]
        pos = self.embedding(pos)  # pos: [seq_len, batch_size, embed_dim]
        x = x + pos #positional embedding -- what the implementation did, but the paper didn't specify
        x = x.permute(1, 0, 2)  # Adjust x to [batch_size, seq_len, embed_dim] for MultiheadAttention
        attn_output, weights = self.multihead_attn(x, x, x) 
        attn_output = attn_output.permute(0, 2, 1) # attn_output: [batch_size, seq_len, embed_dim]
        out1 = self.conv1(attn_output)
        out2 = self.conv2(attn_output)
        combined_cnn_output = torch.cat((out1, out2), dim=1)  # Concatenate along the feature dimension
        # Classification
        out = self.fc(combined_cnn_output)
        if not self.training: #taken from original implementation, to work with the rest of the training script
            return F.softmax(out, dim=-1).max(1)[1], torch.mean(weights, dim=-2)
        return out
    
if __name__ == '__main__':
    # Testing the model works
    x = np.random.randint(0, 255, (10, 20))
    y = np.random.randint(0, 20, (10, 20))
    sam = SAM(num_classes=5, max_len=20)
    out = sam(torch.from_numpy(x).long(), torch.from_numpy(y).long())
    print(out[0])

    sam.eval()

    out, weights = sam(torch.from_numpy(x).long(), torch.from_numpy(y).long())
    print(out[0])
    print("weights shape")
    print(weights.shape)
    print("weights")
    print(weights[0])