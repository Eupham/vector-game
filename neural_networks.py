import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    """RMS normalization without any learned weights."""
    def __init__(self, dim=None, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        # Compute RMS over last dim and normalize
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / rms

class AttentionRegressionNetwork(nn.Module):
    """Vertex-based self-attention network for value prediction."""
    def __init__(self, num_vertices, embed_dim=128, hidden_dim=256, num_layers=2, num_heads=4, batch_size=32):
        super().__init__()
        # Embeddings for vertex position and occupancy
        self.vertex_emb = nn.Embedding(num_vertices, embed_dim)
        self.occupancy_emb = nn.Embedding(3, embed_dim)
        # Multi-head self-attention layers
        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
        # Feed-forward head
        self.ffn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
        # Training batch management
        self.batch_size = batch_size
        self.batch_states_indices = []
        self.batch_states_occupancy = []
        self.batch_targets = []
    
    def forward(self, vertex_indices, occupancy_states):
        # vertex_indices, occupancy_states: (V,) or (B,V)
        if vertex_indices.dim() == 1:
            # single instance -> add batch dim
            vertex_indices = vertex_indices.unsqueeze(0)
            occupancy_states = occupancy_states.unsqueeze(0)
        # Batch of shape (B,V)
        emb = self.vertex_emb(vertex_indices) + self.occupancy_emb(occupancy_states)
        # emb: (B, V, E)
        x = emb
        # Apply stacked self-attention
        for attn in self.attn_layers:
            res, _ = attn(x, x, x)
            x = x + res  # residual connection
        # Global mean pooling over vertices
        x = x.mean(dim=1)  # (B, E)
        # Feed-forward head to predict value
        return self.ffn(x)  # (B, 1)

    # Training step for TD updates
    def train_step(self, vertex_indices, occupancy_states, target, optimizer):
        """Collect single example and optionally run batch training"""
        # Append to batch buffer
        self.batch_states_indices.append(vertex_indices)
        self.batch_states_occupancy.append(occupancy_states)
        self.batch_targets.append(torch.tensor([target], dtype=torch.float32, device=vertex_indices.device))
        # If enough examples, perform batch update
        if len(self.batch_states_indices) >= self.batch_size:
            return self.batch_train_step(optimizer)
        return None

    def batch_train_step(self, optimizer):
        """Train on a batch of collected examples"""
        if not self.batch_states_indices:
            return None
        # Build predictions and targets
        predictions = []
        for idx, occ in zip(self.batch_states_indices, self.batch_states_occupancy):
            pred = self(idx, occ)
            predictions.append(pred)
        # Stack predictions and targets
        if len(predictions) > 1:
            preds = torch.cat(predictions, dim=0).view(-1)
            targets = torch.cat(self.batch_targets, dim=0).view(-1)
        else:
            preds = predictions[0].view(-1)
            targets = self.batch_targets[0].view(-1)
        # Compute MSE loss
        loss = F.mse_loss(preds, targets)
        # Backprop and optimize
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        optimizer.step()
        # Clear buffers
        self.clear_batch()
        return loss.item()

    def clear_batch(self):
        """Clear batch buffers"""
        self.batch_states_indices = []
        self.batch_states_occupancy = []
        self.batch_targets = []

# Alias for existing code
RegressionNetwork = AttentionRegressionNetwork
ValueNetwork = AttentionRegressionNetwork