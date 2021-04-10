"""順伝搬の実装
"""
import numpy as np

# シグモイド-layer
class Sigmoid:
    def __init__(self):
        self.params = []

    def forward(self, x):
        return 1 / (1 + np.exp(-x))

# Affine-layer
class Affine:
    def __init__(self, W, b):
        self.params = [W, b]
    
    def forward(self, x):
        W, b = self.params
        out = np.dot(x, W) + b
        return out

# 2層-layer
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        I, H, O = input_size, hidden_size, output_size

        # 重みとバイアスの初期化
        W1 = np.random.randn(I,H)
        b1 = np.random.randn(H)
        W2 = np.random.randn(H,O)
        b2 = np.random.randn(O)

        # レイヤの生成
        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2)
        ]

        # 全ての重みをリストにまとめる
        self.params = []
        for layer in self.layers:
            self.params += layer.params
        
    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x



def main():
    # TwoLayerNetによるニューラルネットワーク推論
    x = np.random.randn(10,2)
    model = TwoLayerNet(2,4,3)
    s = model.predict(x)
    
    print(f"x: \n{x}")
    print(f"s: \n{s}")

if __name__ == "__main__":
    main()

