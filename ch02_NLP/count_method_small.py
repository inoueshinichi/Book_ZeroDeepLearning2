import sys
sys.path.append("/Users/inoueshinichi/Desktop/DeepLearning2_NLP") # 親ディレクトリのファイルをインポートするための設定
sys.path.append("/home/inoue/MyGithub/DeepLearning2_NLP")
from common.utils import np, preprocess, create_co_matrix, cos_similarity, most_similar, ppmi

import matplotlib.pyplot as plt



def test_count_method_small():
    """特異値分解(SVD)によるPPMI(正の相互情報量行列)の次元削減
    """
    
    text = 'You say goodbye and I say hello.'
    print(text)
    corpus, word_to_id, id_to_word = preprocess(text)
    print(f"corpus: {corpus}")
    print(f"word_to_id: {word_to_id}")
    print(f"id_to_word: {id_to_word}")
    vocab_size = len(word_to_id)
    C = create_co_matrix(corpus, vocab_size)

    # 正の相互情報量
    W = ppmi(C)

    np.set_printoptions(precision=3)
    print("covariance matrix")
    print(C)
    print('-'*50)
    print("PPMI")
    print(W)

    # SVD
    U, S, V = np.linalg.svd(W) # Note@ SVDは行列サイズNに対して、計算時間がO(N^3)とかなり遅いため、通常は、truncated SVD(scikit-learn)を使う
    # UがSVDによって変換された密な単語ベクトル(行列)

    print(f"C[0]: {C[0]}") 
    print(f"W[0]: {W[0]}")
    print(f"U[0]: {U[0]}")

    print(f"U[0, :2]: {U[0, :2]}")


    # 次元削減したベクトルを2次元散布図としてプロット
    for word, word_id in word_to_id.items():
        plt.annotate(word, (U[word_id, 0], U[word_id, 1]))
    
    plt.scatter(U[:, 0], U[:, 1], alpha=0.5)
    plt.show()



if __name__ == "__main__":
    test_count_method_small()