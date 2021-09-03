import sys
sys.path.append("/Users/inoueshinichi/Desktop/MyGithub/DeepLearning2_NLP") # 親ディレクトリのファイルをインポートするための設定
sys.path.append("/home/inoue/MyGithub/DeepLearning2_NLP")

import numpy as np
import matplotlib.pyplot as plt
from dataset import sequence
from common.optimizer import Adam
from common.trainer import Trainer
from common.utils import eval_seq2seq
from peeky_seq2seq import PeekySeq2seq
from seq2seq import Seq2seq


def main():
    
    # データセットの読み込み
    (x_train, t_train), (x_test, t_test) = sequence.load_data('addition.txt')
    char_to_id, id_to_char = sequence.get_vocab()

    # 入力列を逆順にするとSeq2Se2の精度が上がるらしいが。。。クソ理論
    is_reverse = True
    if is_reverse:
        x_train, x_test = x_train[:, ::-1], x_test[:, ::-1]

    # ハイパーパラメータの設定
    vocab_size = len(char_to_id)
    wordvec_size = 16
    hidden_size = 128
    batch_size = 128
    max_epoch = 25
    max_grad = 5.0

    # モデル/オプティマイザ/トレーナーの生成
    # model = Seq2seq(vocab_size, wordvec_size, hidden_size)
    model = PeekySeq2seq(vocab_size, wordvec_size, hidden_size)
    optimizer = Adam()
    trainer = Trainer(model, optimizer)

    acc_list = []
    for epoch in range(max_epoch):
        trainer.fit(x_train, t_train, max_epoch=1,
                    batch_size=batch_size, max_grad=max_grad)

        correct_num = 0
        for i in range(len(x_test)):
            question, correct = x_test[[i]], t_test[[i]]
            verbose = i < 10
            correct_num += eval_seq2seq(model,
                                        question,
                                        correct,
                                        id_to_char,
                                        verbose)
        
        acc = float(correct_num) / len(x_test)
        acc_list.append(acc)
        print(f'val acc {acc * 100}')
    


if __name__ == "__main__":
    main()
