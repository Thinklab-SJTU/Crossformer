import jittor as jt
from jittor import nn
from jittor import nn as F
from cross_models.attn import FullAttention, AttentionLayer, TwoStageAttentionLayer

class DecoderLayer(nn.Module):
    '''
    The decoder layer of Crossformer, each layer will make a prediction at its scale
    '''
    def __init__(self, seg_len, d_model, n_heads, d_ff=None, dropout=0.1, out_seg_num = 10, factor = 10):
        super(DecoderLayer, self).__init__()
        self.self_attention = TwoStageAttentionLayer(out_seg_num, factor, d_model, n_heads, \
                                d_ff, dropout)    
        self.cross_attention = AttentionLayer(d_model, n_heads, dropout = dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_model),
                                nn.GELU(),
                                nn.Linear(d_model, d_model))
        self.linear_pred = nn.Linear(d_model, seg_len)

    def forward(self, x, cross):
        '''
        x:  [B, ts_d, out_seg_num, d_model] —— 上一层 decoder 的输出
        cross: [B, ts_d, in_seg_num, d_model] —— 对应 encoder 层的输出
        '''
        # 记录形状
        B, TS_D, OUT_S, DMODEL = x.shape
        _, _, IN_S, _ = cross.shape

        # 1) 自注意力（Two-Stage）
        x = self.self_attention(x)  # 仍为 [B, ts_d, out_seg_num, d_model]

        # 2) 把 ts_d 合并进 batch，便于后续 AttentionLayer 的 [B, L, D] 接口
        #    'b ts_d L d' -> '(b*ts_d) L d'
        x = x.reshape(B * TS_D, OUT_S, DMODEL)
        cross = cross.reshape(B * TS_D, IN_S, DMODEL)

        # 3) cross-attn
        tmp = self.cross_attention(x, cross, cross)  # 形状 [B*TS_D, OUT_S, DMODEL]
        x = x + self.dropout(tmp)
        y = x = self.norm1(x)
        y = self.MLP1(y)
        dec_output = self.norm2(x + y)               # [B*TS_D, OUT_S, DMODEL]

        # 4) 还原回 [B, ts_d, seg_dec_num, d_model]
        SEG_DEC = dec_output.shape[1]                # 一般等于 OUT_S
        dec_output = dec_output.reshape(B, TS_D, SEG_DEC, DMODEL)

        # 5) 预测：[B, ts_d, seg_dec_num, d_model] -> 线性到 seg_len -> [B, ts_d, seg_dec_num, seg_len]
        layer_predict = self.linear_pred(dec_output)

        # 6) 重排到 [B, (out_d * seg_num), seg_len]
        _, OUT_D, SEG_NUM, SEG_LEN = layer_predict.shape
        layer_predict = layer_predict.reshape(B, OUT_D * SEG_NUM, SEG_LEN)

        return dec_output, layer_predict

    def execute(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Decoder(nn.Module):
    '''
    The decoder of Crossformer, making the final prediction by adding up predictions at each scale
    '''
    def __init__(self, seg_len, d_layers, d_model, n_heads, d_ff, dropout,\
                router=False, out_seg_num = 10, factor=10):
        super(Decoder, self).__init__()

        self.router = router
        self.decode_layers = nn.ModuleList()
        for i in range(d_layers):
            self.decode_layers.append(DecoderLayer(seg_len, d_model, n_heads, d_ff, dropout, \
                                        out_seg_num, factor))

    def forward(self, x, cross):
        # x: [B, ts_d, out_seg_num, d_model]
        final_predict = None
        i = 0

        B, TS_D, _, _ = x.shape

        for layer in self.decode_layers:
            cross_enc = cross[i]            # 对应尺度的 encoder 输出
            x, layer_predict = layer(x, cross_enc)
            if final_predict is None:
                final_predict = layer_predict            # [B, (ts_d*seg_num), seg_len]
            else:
                final_predict = final_predict + layer_predict
            i += 1

        # 原：rearrange(final_predict, 'b (out_d seg_num) seg_len -> b (seg_num seg_len) out_d', out_d=ts_d)
        # 现：先还原出 [B, out_d, seg_num, seg_len] 再换轴折叠
        B, OS, SEG_LEN = final_predict.shape          # OS = out_d * seg_num
        SEG_NUM = OS // TS_D
        final_predict = final_predict.reshape(B, TS_D, SEG_NUM, SEG_LEN)   # [B, out_d, seg_num, seg_len]
        final_predict = final_predict.permute(0, 2, 3, 1)                  # [B, seg_num, seg_len, out_d]
        final_predict = final_predict.reshape(B, SEG_NUM * SEG_LEN, TS_D)  # [B, (seg_num*seg_len), out_d]

        return final_predict
    def execute(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

