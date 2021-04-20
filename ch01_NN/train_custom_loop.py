import sys
sys.path.append("/Users/inoueshinichi/Desktop/DeepLearning2_NLP")
import numpy as np 
from common.optimizer import SGD
from dataset import spiral
import matplotlib.pyplot as plt
from ch01_NN.two_layer_net import TwoLayerNet

def main():
    
    # (1) ハイパーパラメータの設定
    max_epoch = 300
    batch_size = 30
    hidden_size = 10
    learning_rate = 1.0

    # (2) データの読み込み、モデルとオプティマイザの生成
    x, t = spiral.load_data()
    model = TwoLayerNet(input_size=2, hidden_size=hidden_size, output_size=3)
    optimizer = SGD(lr=learning_rate)

    # 学習で使用する変数
    data_size = len(x)
    max_iters = data_size // batch_size
    total_loss = 0
    loss_count = 0
    loss_list = []

    for epoch in range(max_epoch):
        # (3) データのシャッフル
        idx = np.random.permutation(data_size)
        x = x[idx]
        t = t[idx]

        for iters in range(max_iters):
            batch_x = x[iters*batch_size:(iters+1)*batch_size]
            batch_t = t[iters*batch_size:(iters+1)*batch_size]

            # (4) 勾配を求め、パラメータを更新
            loss = model.forward(batch_x, batch_t)
            model.backward()
            optimizer.update(model.params, model.grads)

            total_loss += loss
            loss_count += 1

            # (5) 定期的に学習経過を出力
            if (iters + 1) % 10 == 0:
                avg_loss = total_loss / loss_count
                print(f"| epoch {epoch+1} | iter {iters+1} / {max_iters} | loss {avg_loss}")
                loss_list.append(avg_loss)
                total_loss, loss_count = 0, 0
        
    print(f"loss_list: \n{loss_list}")

    plt.plot(np.arange(len(loss_list)), np.asarray(loss_list))
    plt.xlabel('iterations (x10)')
    plt.ylabel('loss')
    plt.show()

if __name__ == "__main__":
    main()