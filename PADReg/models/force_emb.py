import torch
from torch import nn
import math
import torch.nn.functional as F

class ForceEmbedding(nn.Module):
    def __init__(self, n_channels: int):
        """
        Params:
            n_channels: time_channel
        """
        super().__init__()
        self.n_channels = n_channels
        self.lin1 = nn.Linear(self.n_channels // 4, self.n_channels)
        self.act = nn.SELU()
        self.lin2 = nn.Linear(self.n_channels, self.n_channels)

    def get_emb(self,f,quarter_dim):
        emb = math.log(10_000) / (quarter_dim - 1)
        emb = torch.exp(torch.arange(quarter_dim, device=f.device) * -emb)
        emb = f[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)
        return emb

    def forward(self, f: torch.Tensor):
        """
        Params:
            f: (batch_size x 2), dtype-float32
        Return:
            emb: (batch_size x n_channels), dtype-float32
        """
        quarter_dim = self.n_channels // 16
        emb = torch.cat((self.get_emb(f[:,0],quarter_dim),self.get_emb(f[:,1],quarter_dim)),dim=1)
        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)

        return emb
    
    
class CrossAttention(nn.Module):
    def __init__(self):
        super(CrossAttention, self).__init__()

    def forward(self, query, key, value):
        """
        Forward pass of the cross-attention mechanism.

        Args:
            query (torch.Tensor): Query tensor with shape (bs, 768, 1).
            key (torch.Tensor): Key tensor with shape (bs, 768, 64).
            value (torch.Tensor): Value tensor with shape (bs, 768, 64).

        Returns:
            torch.Tensor: Output tensor with shape (bs, 768, 64).
        """
        # Ensure the input shapes are correct
        bs, d_model, seq_len_q = query.size()
        bs, d_model, seq_len_kv = key.size()

        # Compute the attention scores (query * key^T)
        attention_scores = torch.bmm(query.transpose(1, 2), key)  # Shape: (bs, 1, 64)

        # Apply softmax to get the attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)  # Shape: (bs, 1, 64)

        # Compute the weighted sum of the values
        output = torch.bmm(attention_weights, value.transpose(1, 2))  # Shape: (bs, 1, 768)

        # Transpose back to the original shape
        output = output.transpose(1, 2)  # Shape: (bs, 768, 1)

        # Expand the output to match the shape of the value
        output_expanded = output.expand(-1, -1, seq_len_kv)  # Shape: (bs, 768, 64)

        # Multiply the expanded output with the value
        final_output = output_expanded * value  # Shape: (bs, 768, 64)

        return final_output