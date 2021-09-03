import sys
sys.path.append("/Users/inoueshinichi/Desktop/MyGithub/DeepLearning2_NLP") # 親ディレクトリのファイルをインポートするための設定
sys.path.append("/home/inoue/MyGithub/DeepLearning2_NLP")


from common.optimizer import SGD
from common.trainer import RnnlmTrainer
from common.utils import eval_perplexity
from dataset import ptb
from rnnlm import Rnnlm

def main():
    
    # ハイパーパラメータの設定
    batch_size = 20
    wordvec_size = 100
    hidden_size = 100
    time_size = 35
    lr = 20.0
    max_epoch = 3
    max_grad = 0.25

    # 学習データの読み込み
    corpus, word_to_id, id_to_word = ptb.load_data('train')
    corpus_test, _, _ = ptb.load_data('test')
    vocab_size = len(word_to_id)
    xs = corpus[:-1]
    ts = corpus[1:]

    # モデルの生成
    model = Rnnlm(vocab_size, wordvec_size, hidden_size)
    optimizer = SGD(lr)
    trainer = RnnlmTrainer(model, optimizer)

    # 1) 勾配クリッピングを適用して学習
    trainer.fit(xs, ts, max_epoch, batch_size, time_size, max_grad, eval_interval=20)
    trainer.plot(ylim=(0, 500))

    # 2) テストデータで評価
    model.reset_state()
    ppl_test = eval_perplexity(model, corpus_test)
    print('test perplexity: ', ppl_test)

    # 3) パラメータ保存
    model.save_params()


if __name__ == "__main__":
    main()
