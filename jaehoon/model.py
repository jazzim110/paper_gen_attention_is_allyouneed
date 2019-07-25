from preprocess import Preprocess
import numpy as np
import torch
from torch import nn
import math



#normalize
class Norm(nn.Module):

    def __init__(self, config, logger, eps=1e-6):
        super(Norm, self).__init__()

        self.config = config
        self.logger = logger
        self.size = self.config.dims
        self.alpha = to_cuda(nn.Parameter(torch.ones(self.size)))
        self.bias = to_cuda(nn.Parameter(torch.zeros(self.size)))
        self.eps = eps

    def forward(self, x):

        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias

        return norm

# feed forward network
class Ffn(nn.Module):

    def __init__(self, config, logger):
        super().__init__()

        self.size = config.dims
        self.d_ff = config.d_ff

        self.linear_1 = to_cuda(nn.Linear(self.size, self.d_ff))
        self.dropout = to_cuda(nn.Dropout(config.dropout))
        self.linear_2 = to_cuda(nn.Linear(self.d_ff, self.size))

    def forward(self, x):
        x = self.dropout(nn.functional.relu(self.linear_1(x)))
        x = self.linear_2(x)

        return x



class MultiHeadAttention(nn.Module):

    def __init__(self, config, logger):
        super().__init__()

        self.size = config.dims
        self.heads = config.heads
        self.d_h = self.size // self.heads
        self.batch_size = config.batch_size

        self.q_linear = to_cuda(nn.Linear(self.size, self.size))
        self.k_linear = to_cuda(nn.Linear(self.size, self.size))
        self.v_linear = to_cuda(nn.Linear(self.size, self.size))
        self.dropout = nn.Dropout(config.dropout)
        self.attn_out = to_cuda(nn.Linear(self.size, self.size))

    def forward(self, q, k, v, mask=None):

        # perform linear operation and split into h heads
        q = self.q_linear(q).view(self.batch_size, -1, self.heads, self.d_h)
        k = self.k_linear(k).view(self.batch_size, -1, self.heads, self.d_h)
        v = self.v_linear(v).view(self.batch_size, -1, self.heads, self.d_h)

        # transpose to get dimensions bs * h * sl * d_model
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = self.attention(q, k, v, self.d_h, mask=mask)
        #print(scores.shape)
        #print()

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(self.batch_size, -1, self.size)
        output = self.attn_out(concat)

        return output


    def attention(self, q, k, v, dims_heads, mask=None):

        sm_model = nn.Softmax()
        scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(dims_heads)
        # print(scores.shape)
        # print()

        # need to add mask
        if mask != None:
            masking_mat = torch.ones_like(scores[0, 0, :, :])
            masking_mat_triu = torch.triu(masking_mat, diagonal=1)
            masked_mat = masking_mat_triu * (-1000000000)
            # masked_mat[masked_mat == 0] = 1
            scores += masked_mat  # 더해주면 기존 weight 값은 유지되고, masking한 부분은 마이너스 무한대가 된다.

        scores = sm_model(scores)

        if self.dropout != None:
            scores = self.dropout(scores)

        output = torch.matmul(scores, v)

        return output



class Embedding(nn.Module):

    def __init__(self, config, logger):
        super().__init__()

        self.size = config.dims
        self.dropout = nn.Dropout(config.dropout)
        self.preprocess = Preprocess(config, logger)


    def forward(self, x, cutoff_max_sen_len, vocab_size):

        positional_enc = self.positional_encoding(cutoff_max_sen_len, self.size)
        embedding_table = nn.Embedding(vocab_size, self.size, padding_idx=0)

        input = x.long()
        output = embedding_table(input)

        output = output + positional_enc

        output = self.dropout(output)

        return output


    def positional_encoding(self, cutoff_max_sen_len, dims):

        pos_idx = np.zeros((1, cutoff_max_sen_len))

        for i in range(len(pos_idx)):
            pos_idx[i] = np.arange(cutoff_max_sen_len)

        positional_encoding = list()

        for pos in range(cutoff_max_sen_len):
            if pos == 0:
                positional_encoding.append(np.zeros(dims))
            else:
                pos = [pos / (10000 ** (2 * i / dims)) for i in range(dims)]
                positional_encoding.append(pos)

        positional_encoding = np.array(positional_encoding, dtype=np.float32)
        positional_encoding[1:, 0::2] = np.sin(positional_encoding[1:, 0::2])
        positional_encoding[1:, 1::2] = np.cos(positional_encoding[1:, 1::2])
        positional_encoding = torch.from_numpy(positional_encoding)

        # position_cal.shape : (cutoff_max_sen, dims)
        return positional_encoding



def to_cuda(batch):

    if torch.cuda.is_available():
        batch = batch.cuda()

    return batch


