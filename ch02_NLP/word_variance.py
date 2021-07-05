""" 

    単語の分布仮説
        単語の意味は、周囲の単語によって形成される
        単語自体に意味はなく、コンテキスト(文脈)によって単語の意味が形成される
    
    共起行列
        ある単語に着目した場合、その周囲にどのような単語がどれだけ現れるのかを
        カウントして、集計する.

    You say goodbye and I say hello .
    you: 0
    say: 1
    goodbye: 2
    and: 3
    I: 4
    say: 5
    hello: 6
    .: 7

        you | say | goodbye | and | I | hello | . 
    you   1     0         0     0   0       0   0
    say   1     0         1     0   1       1   0
goodbye   0     1         0     1   0       0   0
    and   0     0         1     0   1       0   0
      I   0     1         0     1   0       0   0
  hello   0     1         0     0   0       0   1
      .   0     0         0     0   0       1   0
"""

import os
import sys
sys.path.append("/Users/inoueshinichi/Desktop/DeepLearning2_NLP")
sys.path.append("/home/inoue/MyGithub/DeepLearning2_NLP")

import numpy as np
from common.utils import preprocess

def word_variance():
    """共起行列の作成
    """
    # -----------
    text = 'You say goodbye and I say hello.'
    corpus, word_to_id, id_to_word = preprocess(text)

    print(f"corpus: {corpus}")
    print(f"id_to_word: {id_to_word}")

    # 共起行列の例
    C = np.array([
        [0, 1, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1, 1, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0]],
        dtype=np.int32)
    print(f"C: {C}")

    print(f"C[0]: {C[0]}") # 単語ID[0]のベクトル
    print(f"C[1]: {C[1]}") # 単語ID[1]のベクトル
    print(C[word_to_id['goodbye']]) # [goodbye]のベクトル

    


if __name__ == "__main__":
    word_variance()