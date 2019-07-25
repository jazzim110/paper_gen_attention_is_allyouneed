from model import *
from model import Norm as Norm
from model import Ffn as Ffn
from model import MultiHeadAttention as MultiHeadAttention



class Encoder(nn.Module):

    def __init__(self, config, logger):
        super(Encoder, self).__init__()

        self.size = config.dims
        self.heads = config.heads
        self.dropout = config.dropout

        self.norm_1 = Norm(config, logger)
        self.norm_2 = Norm(config, logger)

        self.attn = MultiHeadAttention(config, logger)

        self.ffn = Ffn(config, logger)

        self.dropout_1 = nn.Dropout(self.dropout)
        self.dropout_2 = nn.Dropout(self.dropout)

    def forward(self, x):
        x1 = x + self.dropout_1(self.attn(x, x, x))
        x2 = self.norm_1(x1)

        x3 = x2 + self.dropout_2(self.ffn(x2))
        x4 = self.norm_2(x3)

        return x4


class Decoder(nn.Module):

    def __init__(self, config, logger):
        super(Decoder, self).__init__()

        self.size = config.dims
        self.heads = config.heads
        self.dropout = config.dropout

        self.norm_1 = Norm(config, logger)
        self.norm_2 = Norm(config, logger)
        self.norm_3 = Norm(config, logger)

        self.attn_1 = MultiHeadAttention(config, logger)
        self.attn_2 = MultiHeadAttention(config, logger)

        self.ffn = Ffn(config, logger)

        self.dropout_1 = nn.Dropout(self.dropout)
        self.dropout_2 = nn.Dropout(self.dropout)
        self.dropout_3 = nn.Dropout(self.dropout)

    def forward(self, y, e_outputs, f_mask=True, s_mask=None):
        y1 = y + self.dropout_1(self.attn_1(y, y, y, mask=f_mask))
        y2 = self.norm_1(y1)

        y3 = y2 + self.dropout_2(self.attn_2(y2, e_outputs, e_outputs, mask=s_mask))
        y4 = self.norm_2(y3)

        y5 = y4 + self.dropout_3(self.ffn(y4))
        y6 = self.norm_3(y5)

        return y6