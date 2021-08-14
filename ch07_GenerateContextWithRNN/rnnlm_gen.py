import sys
sys.path.append("/Users/inoueshinichi/Desktop/DeepLearning2_NLP") # 親ディレクトリのファイルをインポートするための設定
sys.path.append("/home/inoue/MyGithub/DeepLearning2_NLP")

from ch06_RNNWithGate.rnnlm import Rnnlm
from ch06_RNNWithGate.better_rnnlm import BetterRnnlm

import numpy as np
from common.functions import softmax

class RnnlmGen(Rnnlm):

    def generate(self, start_id, skip_ids=None, sample_size=100):
        word_ids = [start_id]

        x = start_id
        while len(word_ids) < sample_size:
            x = np.array(x).reshape(1, 1)
            score = self.predict(x)
            p = softmax(score.flatten()) # 確率分布 (V, )

            # 与えた確率分布にしたがってサンプリング ※ 0 ~ V-1 を選ぶ
            sampled = np.random.choice(len(p), size=1, p=p)
            
            if (skip_ids is None) or (sampled not in skip_ids):
                x = sampled
                word_ids.append(int(x))

        return word_ids

        