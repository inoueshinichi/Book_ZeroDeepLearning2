import sys
sys.path.append("/Users/inoueshinichi/Desktop/MyGithub/DeepLearning2_NLP") # 親ディレクトリのファイルをインポートするための設定
sys.path.append("/home/inoue/MyGithub/DeepLearning2_NLP")

from dataset.sequence import load_data, get_vocab

def main():
    
    (x_train, t_train), (x_test, t_test) = load_data('addition.txt', seed=1984)
    char_to_id, id_to_char = get_vocab()

    print(x_train.shape, t_train.shape)
    print(x_test.shape, t_test.shape)

    print(''.join([id_to_char[c] for c in x_train[0]]))
    print(''.join([id_to_char[c] for c in t_train[0]]))

if __name__ == "__main__":
    main()
