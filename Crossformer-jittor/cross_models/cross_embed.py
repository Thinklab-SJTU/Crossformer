import jittor as jt
from jittor import nn
import math

class DSW_embedding(nn.Module):
    def __init__(self, seg_len, d_model):
        super(DSW_embedding, self).__init__()
        self.seg_len = seg_len
        self.linear = nn.Linear(seg_len, d_model)

    def forward(self, x):
        # x: [B, L, D]
        B, L, D = x.shape
        seg_num = L // self.seg_len

        # [B, seg_num, seg_len, D]
        x_segment = x.reshape(B, seg_num, self.seg_len, D)
        # [B, D, seg_num, seg_len]
        x_segment = x_segment.permute(0, 3, 1, 2)
        # [(B*D*seg_num), seg_len]
        x_segment = x_segment.reshape(B * D * seg_num, self.seg_len)

        # 线性映射到 d_model
        x_embed = self.linear(x_segment)                         # [(B*D*seg_num), d_model]
        d_model = x_embed.shape[-1]
        # 还原到 [B, D, seg_num, d_model]（等价于原来的 rearrange）
        x_embed = x_embed.reshape(B, D, seg_num, d_model)

        return x_embed

    def execute(self, *args, **kwargs):
        return self.forward(*args, **kwargs)