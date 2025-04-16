class TSM(nn.Module):
    def __init__(self, channels, n_div=8):
        super().__init__()
        self.fold = channels // n_div

    def forward(self, x):
        B, C, T = x.shape
        out = x.clone()
        if T > 5:
            out[:, :self.fold, 5:] += x[:, :self.fold, :-5]
            out[:, self.fold:2*self.fold, :-5] += x[:, self.fold:2*self.fold, 5:]
        return out

class TemporalAttention(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv1d(feat_dim, feat_dim // 4, 1),
            nn.GELU(),
            nn.Conv1d(feat_dim // 4, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attn_weights = self.attn(x)
        return x * attn_weights

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
    def forward(self, x):
        w = F.gelu(self.fc1(x))
        w = torch.sigmoid(self.fc2(w))
        return x * w
class TSMBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_div=8):
        super().__init__()
        self.tsm = TSM(in_channels, n_div)
        self.conv = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.gelu = nn.GELU()
     
        self.attention = TemporalAttention(out_channels)
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        x = self.tsm(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.attention(x)
        return self.gelu(x) + residual

class MemoryModule(nn.Module):
    def __init__(self, feat_dim=256, mem_size=100):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(mem_size, feat_dim))
        self.fc = nn.Linear(feat_dim*2, feat_dim)

    def forward(self, x):
        similarity = torch.mm(F.normalize(x, p=2, dim=1), F.normalize(self.memory, p=2, dim=1).t())
        attn = F.softmax(similarity, dim=1)
        mem_read = torch.mm(attn, self.memory)
        return self.fc(torch.cat([x, mem_read], dim=1))

class TSMAE(nn.Module):
    def __init__(self, in_channels=3, n_div=8):
        super().__init__()

        # Encoder with TSM Blocks and pooling for efficiency.
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 64, 3, padding=1),
            TSMBlock(64, 64, n_div),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),
            TSMBlock(64, 128, n_div),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.se = SEBlock(128)

        
        self.memory = MemoryModule(128)
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Linear(256, 3 * 16) 
        )

        
        self.scorer = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
       
        encoded = self.encoder(x).squeeze(-1) 
        encoded = self.se(encoded)
        mem_features = self.memory(encoded)
        recon = self.decoder(mem_features).view(-1, 3, 16)

        score = self.scorer(mem_features).squeeze()
        return score, recon, encoded
