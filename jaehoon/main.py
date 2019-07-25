from layer_stack import Transformer
from preprocess import Preprocess
from model import Embedding
from config import get_args
from logger import set_logger
from tqdm import tqdm_notebook
from torch import nn
import torch
import time



def main():

    config = get_args()
    logger = set_logger(config)

    batch_size = config.batch_size
    preprocess = Preprocess(config, logger)

    # preprocess
    token2idx_en, idx2token_en, token2idx_de, idx2token_de, inputs, outputs, cutoff_max_sen_len = \
        preprocess.get_final_data()

    # data split
    input_train = inputs[:int(len(inputs)*0.8)]
    output_train = outputs[:int(len(outputs)*0.8)]

    input_test = inputs[int(len(inputs)*0.8):]
    output_test = outputs[int(len(outputs)*0.8):]

    source_train = preprocess.dataloader(input_train, output_train, batch_size)
    source_test = preprocess.dataloader(input_test, output_test, batch_size)

    # training
    train_model_path = train(config, logger, source_train, cutoff_max_sen_len)
    logger.info(print("train model path : {}".format(train_model_path)))



def train(config, logger, source_train, cutoff_max_sen_len):

    lr = config.learning_rate
    epochs = config.n_epochs
    vocab_size = config.vocab_size
    print_fre = config.print_fre

    embedding = Embedding(config, logger)

    model = Transformer(config, logger)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()

    start = time.time()
    temp = start
    total_loss = 0

    #total = tqdm_notebook(source)

    for epoch in range(epochs):

        for idx, (x, y) in enumerate(source_train):

            src = embedding(x, cutoff_max_sen_len, vocab_size)
            trg = embedding(y, cutoff_max_sen_len, vocab_size)
            label = torch.tensor(torch.reshape(y, (-1,)), dtype=torch.int64)

            src = to_cuda(src)
            trg = to_cuda(trg)
            label = to_cuda(label)

            pred = model(src, trg, cutoff_max_sen_len, vocab_size)

            LABEL = label.cpu().numpy()
            PRED = pred.cpu().detach().numpy()

            opt.zero_grad()
            loss = nn.functional.cross_entropy(input=pred.view(-1, pred.size(-1)),
                                               target=label, ignore_index=0)
            loss.backward()

            opt.step()

            total_loss += loss.data
            if (idx + 1) % print_fre == 0:
                loss_avg = total_loss / print_fre
                print("time = %dm, epoch %d, iter = %d, loss = %.3f, %ds per %d iters"
                      % ((time.time() - start) // 60, epoch+1, idx+1, loss_avg, time.time()-temp, print_fre))
                total_loss = 0
                temp = time.time()
                logger.info("time = %dm, epoch %d, iter = %d, loss = %.3f, %ds per %d iters"
                            % ((time.time() - start) // 60, epoch+1, idx+1, loss_avg, time.time()-temp, print_fre))

        model_name = str(epoch+1) + "-" + str(loss_avg)
        model_path = config.model_dir + model_name
        torch.save(model.state_dict(), model_path)

    print("training is finished ")

    return model_path



def to_cuda(batch):
    if torch.cuda.is_available():
        batch = batch.cuda()

    return batch






if __name__ == "__main__":
    main()




