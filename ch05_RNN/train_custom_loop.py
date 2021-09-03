import sys
sys.path.append("/Users/inoueshinichi/Desktop/MyGithub/DeepLearning2_NLP") # 親ディレクトリのファイルをインポートするための設定
sys.path.append("/home/inoue/MyGithub/DeepLearning2_NLP")

import matplotlib.pyplot as plt
import numpy as np
from common.optimizer import SGD
from dataset import ptb
from simple_rnnlm import SimpleRnnlm


def main():
    # ハイパーパラメータの設定
    batch_size = 10
    wordvec_size  = 100
    hidden_size  = 500
    time_size = 5 # Truncated BPTTの展開する時間サイズ
    lr = 0.1
    max_epoch = 100

    # 学習データセットの読み込み
    corpus, word_to_id, id_to_word = ptb.load_data('train')
    corpus_size  =1000
    corpus = corpus[:corpus_size]
    vocab_size = int(max(corpus) + 1)

    print("corpus: ", corpus)
    print("corpus size: ", len(corpus))

    # 入力
    xs = corpus[:-1]
    ts = corpus[1:] # 教師ラベル = 入力の次の単語ID
    print("corpus size: %d, vocabulary size: %d" % (corpus_size, vocab_size))
    data_size = len(xs)
    print("data_size: ", data_size)

    # 学習時に使用する変数
    max_iters = data_size // (batch_size * time_size)
    time_idx = 0
    total_loss = 0
    loss_count  =0
    ppl_list = []

    # モデルの生成
    model = SimpleRnnlm(vocab_size, wordvec_size, hidden_size)
    optimizer = SGD(lr)

    # (1) ミニバッチの各サンプルの読み込み開始位置を計算
    jump = (corpus_size - 1) // batch_size # e.g. corpus_size=1000, batch_size=20 -> (1000-1)//20 -> 49
    print("jump: ", jump)
    offsets = [i * jump for i in range(batch_size)] # batch_size=20, jump=49 -> [0, 49, 98, ..., 931]
    print("offsets: ", offsets)

    # 学習
    for epoch in range(max_epoch):
        for iter in range(max_iters):

            # (2) ミニバッチの取得
            batch_x = np.empty((batch_size, time_size), dtype=np.int32)
            batch_t = np.empty((batch_size, time_size), dtype=np.int32)
            for t in range(time_size):
                for i, offset in enumerate(offsets):
                    print(f"offset: {offset}, time_idx: {time_idx} -> {(offset + time_idx) % data_size}")
                    batch_x[i, t] = xs[(offset + time_idx) % data_size]
                    batch_t[i, t] = ts[(offset + time_idx) % data_size]
                time_idx += 1 # 0リセットされないので、iterループ毎にcorpusのoffsetからのズレを変更している.
            
            # 勾配を求め、パラメータを更新
            loss = model.forward(batch_x, batch_t)
            model.backward()
            optimizer.update(model.params, model.grads)
            total_loss += loss
            loss_count += 1

        # (3) エポック毎にパープレキシティの評価
        ppl = np.exp(total_loss / loss_count)
        print('| epoch %d | perplexity %.2f' % (epoch+1, ppl))
        ppl_list.append(float(ppl))
        total_loss, loss_count = 0, 0

    # グラフの描画
    x = np.arange(len(ppl_list))
    plt.plot(x, ppl_list, label='train')
    plt.xlabel('epochs')
    plt.ylabel('perplexity')
    plt.show()     



if __name__ == "__main__":
    main()