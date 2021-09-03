import sys
sys.path.append("/Users/inoueshinichi/Desktop/MyGithub/DeepLearning2_NLP") # 親ディレクトリのファイルをインポートするための設定
sys.path.append("/home/inoue/MyGithub/DeepLearning2_NLP")

import numpy as np


from common.trainer import Trainer
from common.optimizer import Adam
from ch03_Word2Vec.simple_cbow import SimpleCBOW
from common.utils import preprocess, create_context_target, convert_one_hot

def test_train_word2vec_model():
    """word2vecモデルの学習
    """

    window_size = 1
    hidden_size = 5 # 単語の分散表現ベクトルの次元数
    batch_size = 3
    max_epoch = 1000

    text = 'You say goodbye and I say hello.'

    # コーパスの作成
    corpus, word_to_id, id_to_word = preprocess(text)

    # コンテキストとターゲットの作成
    vocab_size = len(word_to_id)
    contexts, target = create_context_target(corpus, window_size)
    target = convert_one_hot(target, vocab_size)
    contexts = convert_one_hot(contexts, vocab_size)
    print("one-hot target: ", target)
    print("one-hot contexts: ", contexts)

    # CBOWモデル
    model = SimpleCBOW(vocab_size, hidden_size)
    optimizer = Adam()

    # trainer
    trainer = Trainer(model, optimizer)

    # 学習
    trainer.fit(contexts, target, max_epoch=max_epoch, batch_size=batch_size)
    trainer.plot()

    # CBOWの重み(W_in)を取得する
    word_vecs = model.word_vecs
    for word_id, word in id_to_word.items():
        print(word, word_vecs[word_id])
    

if __name__ == "__main__":
    test_train_word2vec_model()
