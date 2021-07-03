import sys
sys.path.append("/Users/inoueshinichi/Desktop/DeepLearning2_NLP")
sys.path.append("/home/inoue/Desktop/DeepLearning2_NLP")
import os
from common.np import *

from typing import (
    NoReturn,
    Union,
    List,
    Dict,
    Tuple,
    Callable
)

def clip_grads(grads: np.ndarray, max_norm: float):
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad**2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:
        for grad in grads:
            grad *= rate


def preprocess(text: str) -> Tuple[np.ndarray, Dict, Dict]:
    """コーパスを作る前処理

    Args:
        text (str): [description]

    Returns:
        Tuple[np.ndarray, Dict, Dict]: [description]
    """
    text = text.lower()
    text = text.replace('.', ' .')
    # print(f"text: {text}")

    words = text.split(' ')
    # print(f"words: {words}")

    # ------------
    word_to_id = {}
    id_to_word = {}

    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    # print(f"word_to_id: {word_to_id}")
    # print(f"id_to_word: {id_to_word}")

    # import numpy as np
    corpus = [word_to_id[w] for w in words]
    corpus = np.array(corpus)
    # print(f"corpus: {corpus}")

    return corpus, word_to_id, id_to_word


def create_co_matrix(corpus: List[int], vocab_size: int, window_size: int = 1):
    """共起行列の作成

    Args:
        corpus (List[int]): [description]
        vocab_size (int): [description]
        window_size (int, optional): [description]. Defaults to 1.
    """
    

