from jaehoon.preprocess import Preprocess
from jaehoon.config import get_args
from jaehoon.logger import set_logger



config = get_args()
logger = set_logger(config)
preprocess = Preprocess(config, logger)
X, Y, token2idx_en, idx2token_en, token2idx_de, idx2token_de = preprocess.preprocessing()

for x, y in zip(X, Y):
    print(x.shape)
    print(y.shape)

    logger.info("x, y shape")
    logger.info(x.shape)
    logger.info(y.shape)
    break