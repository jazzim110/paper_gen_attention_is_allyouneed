from en_de_layer import Encoder, Decoder
from model import Embedding
from torch import nn
import torch


class Encoder_layers(nn.Module):

    def __init__(self, config, logger):
        super(Encoder_layers, self).__init__()

        self.N = config.N_turns
        self.size = config.dims

        self.encoder = Encoder(config, logger)
        self.layers = get_clones(self.encoder, self.N)

    def forward(self, src, cutoff_max_sen_len, vocab_size):

        for i in range(self.N):
            src_output = self.layers[i](src)

        return src_output



class Decoder_layers(nn.Module):

    def __init__(self, config, logger):
        super(Decoder_layers, self).__init__()

        self.config = config
        self.logger = logger
        self.N = self.config.N_turns
        self.size = self.config.dims

        self.layers = get_clones(Decoder(config, logger), self.N)

    def forward(self, trg, cutoff_max_sen_len, vocab_size, e_outputs, f_mask=True, s_mask=None):

        for i in range(self.N):
            trg_output = self.layers[i](trg, e_outputs, f_mask, s_mask )

        return trg_output



class Transformer(nn.Module):
    def __init__(self, config, logger):
        super(Transformer, self).__init__()

        self.size = config.dims
        self.vocab_size = config.vocab_size

        self.softmax = to_cuda(nn.Softmax())
        self.out = to_cuda(nn.Linear(self.size, self.vocab_size))
        self.encoder_stack = Encoder_layers(config, logger)
        self.decoder_stack = Decoder_layers(config, logger)


    def forward(self, x, y, cutoff_max_sen_len, vocab_size):
        e_outputs = self.encoder_stack(x, cutoff_max_sen_len, vocab_size)
        d_outputs = self.decoder_stack(y, cutoff_max_sen_len, vocab_size, e_outputs)
        output = self.softmax(self.out(d_outputs))

        return output



def get_clones(module, N):

    return nn.ModuleList([module for i in range(N)])



def to_cuda(batch):

    if torch.cuda.is_available():
        batch = batch.cuda()

    return batch

