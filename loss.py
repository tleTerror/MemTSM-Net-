class FocalLoss(nn.Module):
    def __init__(self, alpha=0.85, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return focal_loss.mean()

class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.focal = FocalLoss(alpha=0.85, gamma=2.0)  
        self.recon_loss = nn.MSELoss()

    def forward(self, scores, recon, encoded, targets, inputs):
        rec_target = inputs[:, :, 8:24]
        rec_loss = self.recon_loss(recon, rec_target)

        # Classification loss
        cls_loss = self.focal(scores, targets)

        # Memory regularization
        mem_reg = torch.mean(torch.mm(encoded, encoded.t()))

        return cls_loss + 0.5 * rec_loss + 0.01 * mem_reg
