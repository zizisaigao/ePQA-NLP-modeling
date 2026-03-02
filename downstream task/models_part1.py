import torch
import torch.nn as nn


class RNNLM(nn.Module):
    """
    Simple RNN language model (Part1-style).
    - Embedding -> RNN -> Linear vocab projection
    - Implements encode() returning [B,T,H] for downstream tasks.
    """
    def __init__(self, vocab_size: int, emb_dim: int = 300, hid_dim: int = 300, num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.RNN(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=num_layers,
            dropout=(dropout if num_layers > 1 else 0.0),
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hid_dim, vocab_size)
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.num_layers = num_layers

    def encode(self, input_ids: torch.Tensor, attention_mask=None) -> torch.Tensor:
        """
        Part3 expects input_ids as [B,T]. Returns [B,T,H].
        """
        if input_ids.dim() != 2:
            raise ValueError(f"encode expects 2D tensor, got {input_ids.shape}")

        x = input_ids.transpose(0, 1).contiguous()  # [T,B]
        emb = self.dropout(self.tok_emb(x))         # [T,B,E]
        out, _ = self.rnn(emb)                      # [T,B,H]
        return out.transpose(0, 1).contiguous()     # [B,T,H]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [T,B] (Part1 LM training convention)
        returns logits: [T,B,V]
        """
        T, B = x.size()
        emb = self.dropout(self.tok_emb(x))          # [T,B,E]
        out, _ = self.rnn(emb)                       # [T,B,H]
        logits = self.fc(out)                        # [T,B,V]
        return logits


class LSTMLM(nn.Module):
    """
    Simple LSTM language model (Part1-style).
    - Embedding -> LSTM -> Linear vocab projection
    - Implements encode() returning [B,T,H] for downstream tasks.
    """
    def __init__(self, vocab_size: int, emb_dim: int = 300, hid_dim: int = 300, num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=num_layers,
            dropout=(dropout if num_layers > 1 else 0.0),
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hid_dim, vocab_size)
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.num_layers = num_layers

    def encode(self, input_ids: torch.Tensor, attention_mask=None) -> torch.Tensor:
        """
        Part3 expects input_ids as [B,T]. Returns [B,T,H].
        """
        if input_ids.dim() != 2:
            raise ValueError(f"encode expects 2D tensor, got {input_ids.shape}")

        x = input_ids.transpose(0, 1).contiguous()  # [T,B]
        emb = self.dropout(self.tok_emb(x))         # [T,B,E]
        out, _ = self.lstm(emb)                     # [T,B,H]
        return out.transpose(0, 1).contiguous()     # [B,T,H]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [T,B]
        returns logits: [T,B,V]
        """
        emb = self.dropout(self.tok_emb(x))          # [T,B,E]
        out, _ = self.lstm(emb)                      # [T,B,H]
        logits = self.fc(out)                        # [T,B,V]
        return logits


class TransformerLM(nn.Module):
    """
    Decoder-only Transformer LM (your Part1/2 version).
    - Embedding + Positional embedding -> TransformerEncoder with causal mask -> Linear vocab projection
    - Implements encode() returning [B,T,D] for downstream tasks.
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 300,
        nhead: int = 4,
        num_layers: int = 4,
        dim_ff: int = 1024,
        dropout: float = 0.2,
        max_len: int = 2048,
    ):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=False,
        )
        self.tr = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.max_len = max_len
        self.d_model = d_model

    def encode(self, input_ids: torch.Tensor, attention_mask=None) -> torch.Tensor:
        """
        Part3 expects input_ids as [B,T]. Returns [B,T,D].
        """
        if input_ids.dim() != 2:
            raise ValueError(f"encode expects 2D tensor, got {input_ids.shape}")

        x = input_ids.transpose(0, 1).contiguous()  # [T,B]
        T, B = x.size()
        pos = torch.arange(0, T, device=x.device).unsqueeze(1).expand(T, B)
        h = self.tok_emb(x) + self.pos_emb(pos)
        h = self.dropout(h)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        out = self.tr(h, mask=mask)                 # [T,B,D]
        return out.transpose(0, 1).contiguous()     # [B,T,D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [T,B]
        returns logits: [T,B,V]
        """
        T, B = x.size()
        pos = torch.arange(0, T, device=x.device).unsqueeze(1).expand(T, B)
        h = self.tok_emb(x) + self.pos_emb(pos)
        h = self.dropout(h)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        out = self.tr(h, mask=mask)
        logits = self.fc(out)
        return logits
