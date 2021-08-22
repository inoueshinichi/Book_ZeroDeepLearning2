import sys
sys.path.append("/Users/inoueshinichi/Desktop/DeepLearning2_NLP") # 親ディレクトリのファイルをインポートするための設定
sys.path.append("/home/inoue/MyGithub/DeepLearning2_NLP")

from rnnlm_gen import RnnlmGen
from dataset import ptb

import numpy as np

def main():
    curpus, word_to_id, id_to_word = ptb.load_data('train')
    vocab_size = len(word_to_id)
    corpus_size = len(curpus)

    model = RnnlmGen()
    # model.load_params('../ch06_RNNWithGate/Rnnlm.pkl')

    # start文字とskip文字の設定
    start_word = 'you'
    start_id = word_to_id[start_word]
    skip_words = ['N', '<unk>', '$']
    skip_ids = [word_to_id[w] for w in skip_words]

    # 文書生成
    word_ids = 
    

if __name__ == "__main__":
    main()

