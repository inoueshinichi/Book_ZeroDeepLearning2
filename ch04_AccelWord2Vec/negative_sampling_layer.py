import sys
sys.path.append("/Users/inoueshinichi/Desktop/DeepLearning2_NLP") # 親ディレクトリのファイルをインポートするための設定
sys.path.append("/home/inoue/MyGithub/DeepLearning2_NLP")

from common.np import *  # import numpy as np
from common.layers import Embedding, SigmoidWithLoss

import collections

# EmbeddingDot: 多値分類を2値分類近似に対応させたEmbeddingレイヤ
class EmbeddingDot:

    def __init__(self, W):
        self.embed = Embedding(W)
        self.params = self.embed.params
        self.grads = self.embed.grads
        self.cache = None

    def forward(self, h, idx):
        target_W = self.embed.forward(idx) # (N, H) = (指定した単語ID数, 内部の重みの次元)
        out = np.sum(target_W * h, axis=1) # (N, H) * (N, H) -> (N, H) -> (N, 1)
        self.cache = (h, target_W)
        return out

    def backward(self, dout):
        h, target_W = self.cache

        dout = dout.reshape(dout.shape[0], 1)

        # target_Wの勾配
        dtarget_W = dout * h
        self.embed.backward(dtarget_W)
        
        # 下流に伝搬させる勾配
        dh = dout * target_W
        return dh

    
# ネガティブサンプリングレイヤの実装
class UnigramSampler:

    def __init__(self, corpus, power, sample_size):
        self.sample_size = sample_size
        self.vocab_size = None
        self.word_p = None

        # 単語の出現回数を取得
        counts = collections.Counter()
        for word_id in corpus:
            counts[word_id] += 1 # 昇順(0, 1, 2, 3, ...)に頻度が格納される
        
        self.vocab_size = len(counts)
        self.word_p = np.zeros(self.vocab_size)

        for i in range(self.vocab_size):
            self.word_p[i] = counts[i]
        
        self.word_p = np.power(self.word_p, power)
        self.word_p /= np.sum(self.word_p)


    def get_negative_sample(self, target):

        batch_size = target.shape[0]

        if not GPU:
            negative_sample = np.zeros((batch_size, self.sample_size), dtype=np.int32)
            for i in range(batch_size):
                p = np.copy(self.word_p) # 0:@, 1:@, 2:@, 3:@, 4:@, ...
                target_idx = target[i]
                p[target_idx] = 0
                p /= np.sum(p) # ターゲットを省いた場合の確率分布
                negative_sample[i, :] = np.random.choice(self.vocab_size, size=self.sample_size, replace=False, p=p)
        else:
            # GPU(cupy）で計算するときは、速度を優先
            # 負例にターゲットが含まれるケースがある
            negative_sample = np.random.choice(self.vocab_size, size=(batch_size, self.sample_size),
                                               replace=True, p=self.word_p)

        return negative_sample
    

# Nagative Samplingの実装
class NegativeSamplingLoss:

    def __init__(self, W, corpus, power=0.75, sample_size=5):
        self.sample_size = sample_size
        self.sampler = UnigramSampler(corpus, power, sample_size)
        self.loss_layers = [SigmoidWithLoss() for _ in range(sample_size + 1)]
        self.embed_dot_layers = [EmbeddingDot(W) for _ in range(sample_size + 1)]

        self.params, self.grads = [], []
        for layer in self.embed_dot_layers:
            self.params += layer.params
            self.grads += layer.grads
        

    def forward(self, h, target):
        batch_size = target.shape[0]
        negative_sample = self.sampler.get_negative_sample(target)

        # 正例のフォワード
        score = self.embed_dot_layers[0].forward(h, target)
        correct_label = np.ones(batch_size, dtype=np.int32)
        loss = self.loss_layers[0].forward(score, correct_label)

        # 負例のフォワード
        negatibe_label = np.zeros(batch_size, dtype=np.int32)
        for i in range(self.sample_size):
            negative_target = negative_sample[:, i]
            score = self.embed_dot_layers[1 + i].forward(h, negative_target)
            loss += self.loss_layers[1 + i].forward(score, negative_label)
        
        return loss

    
    def backward(self, dout=1):
        dh = 0
        for l0, l1 in zip(self.loss_layers, self.embed_dot_layers):
            dscore = l0.backward(dout)
            dh += l1.backward(dscore)
        return dh
    

