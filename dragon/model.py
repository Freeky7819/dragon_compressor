import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ResonantPointer(nn.Module):
    """Basic unit for finding importance."""
    def __init__(self, d_model: int, n_heads: int = 8, depth: int = 2, dropout: float = 0.05):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(d_model)
        self.final = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.norm(x)
        x = self.transformer(x)
        return self.final(x).squeeze(-1)

class MultiPhaseResonantPointer(nn.Module):
    """
    Advanced pointer with multiple phases and LSTM memory.
    This is the 'Phase II' logic that your model expects.
    """
    def __init__(self, d_model: int, n_phases: int = 2, total_depth: int = 4, dropout: float = 0.05):
        super().__init__()
        depth_per_phase = max(1, total_depth // n_phases)
        
        # Pointer phases
        self.phases = nn.ModuleList([
            ResonantPointer(d_model=d_model, depth=depth_per_phase, dropout=dropout)
            for _ in range(n_phases)
        ])
        
        # --- MISSING PARTS (that caused the error) ---
        # Bottleneck summary for phase memory
        self.phase_projector = nn.Linear(d_model, d_model // 2)
        self.phase_memory = nn.LSTM(
            input_size=d_model // 2,
            hidden_size=d_model,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )

        # Confidence gate
        self.confidence_gate = nn.Linear(d_model, 1)
        self.phase_weights = nn.Parameter(torch.ones(n_phases) / n_phases)
        
        # Residual feedback strength
        self.residual_alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, hidden):
        B, T, D = hidden.shape
        accumulated_logits = torch.zeros(B, T, device=hidden.device, dtype=hidden.dtype)
        
        memory_state = None
        current_hidden = hidden

        # Normalized weights
        weights_raw = F.softplus(self.phase_weights)
        weights = weights_raw / (weights_raw.sum() + 1e-6)
        
        for i, pointer in enumerate(self.phases):
            phase_scores = pointer(current_hidden)
            
            # Confidence
            gate_raw = self.confidence_gate(current_hidden)
            confidence = torch.sigmoid(gate_raw.squeeze(-1) * 8.0)
            
            weight = weights[i]
            weighted_scores = phase_scores * confidence * weight
            accumulated_logits = accumulated_logits + weighted_scores
            
            # Memory Feedback (This is the essence of your model)
            # For inference we don't necessarily need noise, but the logic must remain
            summary = current_hidden
            summary = self.phase_projector(summary.mean(dim=1, keepdim=True))
            lstm_out, memory_state = self.phase_memory(summary, memory_state if i > 0 else None)
            
            feedback = lstm_out.expand(-1, T, -1)
            current_hidden = hidden + self.residual_alpha * feedback
            
        return accumulated_logits

class DragonArchitecture(nn.Module):
    """Main architecture for compression."""
    def __init__(self, d_model=384, max_seq_len=128):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # 1. Pointer
        self.pointer = MultiPhaseResonantPointer(d_model=d_model, n_phases=2, total_depth=4)
        
        # 2. Mixer (Soft Merge neighbors)
        self.neighbor_mixer = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model//32),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=2, dilation=2, groups=d_model//32),
        )
        
        # 3. Reconstruction
        self.residual = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
        )
        self.pos_bias = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        
        self.harmonic_w = nn.Parameter(torch.tensor(0.7))
        self.gamma = 0.0025
        
        # --- ADDED: LayerNorm required by your .pth ---
        # Apparently it was defined in the original architecture, although possibly unused
        self.ln = nn.LayerNorm(d_model)

    def harmonic_injection(self, x):
        B, T, D = x.shape
        pos = torch.arange(T, device=x.device).float()
        
        # --- CHANGE HERE (6.28 -> 6.0) ---
        # Your hypothesis: w=6 creates a better interference pattern
        sig = torch.exp(-self.gamma * pos) * torch.sin(6.0 * pos + math.pi/3)
        # -------------------------------------
        
        return x + self.harmonic_w * sig.unsqueeze(0).unsqueeze(-1)

    def compress(self, x, ratio: int = 16):
        B, T, D = x.shape
        k = max(1, T // ratio)
        
        # 1. Position injection
        h = self.harmonic_injection(x)
        
        # 2. Pointer selects important points
        logits = self.pointer(h)
        vals, top_indices = logits.topk(k, dim=1)
        
        # 3. Mixer captures context
        m = self.neighbor_mixer(h.transpose(1,2)).transpose(1,2)
        compressed = m.gather(1, top_indices.unsqueeze(-1).expand(-1,-1, D))
        
        # 4. Gate (filtering)
        gate = torch.sigmoid(vals).unsqueeze(-1)
        compressed = compressed * gate
        
        # --- FIX: NORMALIZATION WAS MISSING HERE ---
        compressed = self.ln(compressed)
        # -------------------------------------------------
        
        # Normalized positions for storage
        norm_positions = top_indices.float() / self.max_seq_len
        
        return compressed, norm_positions

    def decompress(self, compressed, norm_positions, original_T=None):
        if original_T is None: original_T = self.max_seq_len
        B, K, D = compressed.shape
        
        summary = compressed.mean(1)
        background = self.residual(summary).unsqueeze(1).expand(-1, original_T, -1)
        
        if original_T == self.max_seq_len:
            background = background + self.pos_bias
        
        recon = background.clone()
        idx = (norm_positions * original_T).long().clamp(0, original_T-1)
        recon.scatter_(1, idx.unsqueeze(-1).expand(-1, -1, D), compressed)
        return recon