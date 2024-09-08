#Temporal Attention Mechanism
class TemporalAttention(nn.Module):
    def __init__(self, num_channels, num_time_steps):
        super(TemporalAttention, self).__init__()
        self.U1 = nn.Parameter(torch.randn(num_channels, num_time_steps))
        self.U2 = nn.Parameter(torch.randn(num_time_steps, num_time_steps))
        self.U3 = nn.Parameter(torch.randn(num_time_steps, num_time_steps))
        self.be = nn.Parameter(torch.randn(1, num_time_steps, num_time_steps))
        self.Ve = nn.Parameter(torch.randn(num_channels, num_time_steps))

    def forward(self, X):
        # X shape: (batch_size, num_channels, num_time_steps)
        temporal_attention = torch.matmul(X.transpose(-1, -2), self.U1)
        temporal_attention = torch.matmul(temporal_attention, self.U2)
        temporal_attention = torch.matmul(temporal_attention, self.U3.transpose(-1, -2))
        temporal_attention = temporal_attention + self.be
        temporal_attention = F.softmax(temporal_attention, dim=-1)
        temporal_attention = torch.matmul(self.Ve, temporal_attention)
        return temporal_attention
