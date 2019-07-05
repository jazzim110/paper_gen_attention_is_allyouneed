from jaehoon.preprocess import Preprocess
from jaehoon.config import get_args


config = get_args()
preprocess = Preprocess(config)
X, Y, token2idx_en, idx2token_en, token2idx_de, idx2token_de = preprocess.preprocessing()

for x in X:
    print(x.shape)
    break