import sys
sys.path.append("/Users/inoueshinichi/Desktop/DeepLearning2_NLP") # 親ディレクトリのファイルをインポートするための設定
sys.path.append("/home/inoue/MyGithub/DeepLearning2_NLP")

import numpy as np
import matplotlib.pyplot as plt
from dataset import sequence
from common.optimizer import Adam
from common.trainer import Trainer
from common.utils import eval_seq2seq
# from peeky_seq2seq import PeekySeq2Seq


def main():
    
    # データセットの読み込み
    (x_train, t_train), (x_test, t_test) = sequence.load_data('addition.txt')
    char_to_id, id_to_char = sequence.get_vocab()

    # ハイパーパラメータの設定
    vocab_size = len(char_to_id)
    wordvec_size = 16
    hidden_size = 128
    batch_size = 128
    max_epoch = 25
    max_grad = 5.0

    # モデル/オプティマイザ/トレーナーの生成
    model = Seq2seq(vocab_size, wordvec_size, hideen_size)
    optimizer = Adam()
    trainer = Trainer(model, optimizer)

    acc_list = []
    for epoch in range(max_epoch):
        trainer.fit()

if __name__ == "__main__":
    main()
