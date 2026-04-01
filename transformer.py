import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F



class LongSequenceClassifier(nn.Module):
    def __init__(self, input_dim=14, embed_dim=128, nhead=8, ff_dim=512, num_layers=2, num_classes=2):

        super().__init__()
        
        
        
        
        
        """
        # Conv1D per estrazione feature locale + downsampling
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, stride=8, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, embed_dim, kernel_size=3, stride=8, padding=1),
            nn.ReLU()
        )
        """

        
        self.project = nn.Linear(input_dim, embed_dim)
        self.project_q = nn.Linear(3, embed_dim)
        self.project_v = nn.Linear(11, embed_dim)
        self.project_k = nn.Linear(11,embed_dim)# k e v insieme
        # Positional encoding parametrico
        self.pos_enc = nn.Parameter(torch.randn(1, 1024, embed_dim)) 
        
          
        # Transformer encoder
        self.transformer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            batch_first=True
        )
      
        #self.transformer = nn.MultiheadAttention(embed_dim, nhead, batch_first=True)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        """
        x: [batch, seq_len, input_dim]
        """
        # Conv1D richiede [batch, channels, seq_len]
        x = x.permute(0, 2, 1)  # [B, input_dim, seq_len]
        x = F.adaptive_avg_pool1d(x, 1024)
        x = x.permute(0, 2, 1)  # [B, seq_len_down, embed_dim]
        
        #coord = x[:,:,:3]
        #feat = x[:,:,3:]
        
        x = self.project(x)
        #q = self.project_q(coord)
        
        #k= self.project_k(feat) 
        #v = self.project_v(feat)
        
        # Positional encoding
        x = x + self.pos_enc[:, :x.size(1), :]
        
        # Transformer
        x = self.transformer(x)  # [B, seq_len_down, embed_dim]
        
        # Global average pooling lungo la sequenza
        x = x.mean(dim=1)  # [B, embed_dim]
        
        # Classification head
        logits = self.classifier(x)  # [B, num_classes]
        return logits
