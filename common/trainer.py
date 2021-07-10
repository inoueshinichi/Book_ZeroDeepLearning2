import sys
sys.path.append("/Users/inoueshinichi/Desktop/DeepLearning2_NLP")
sys.path.append("/home/inoue/Desktop/DeepLearning2_NLP")
import numpy
import time
import matplotlib.pyplot as plt
from common.np import * # import numpy as np
from common.utils import clip_grads


class Trainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.loss_list = []
        self.eval_interval = None
        self.current_epoch = 0

    def fit(self, x, t, 
            max_epoch=10, batch_size=32, 
            max_grad=None, eval_interval=20):
        
        data_size = len(x)
        max_iters = data_size // batch_size
        self.eval_interval = eval_interval
        model, optimizer = self.model, self.optimizer
        total_loss = 0
        loss_count = 0

        start_time = time.time()
        for epoch in range(max_epoch):
            # シャッフル
            idx = numpy.random.permutation(numpy.arange(data_size))
            x = x[idx]
            y = t[idx]

            for iters in range(max_iters):
                batch_x = x[iters*batch_size:(iters+1)*batch_size]
                batch_t = t[iters*batch_size:(iters+1)*batch_size]
                # print(f"batch_x shape: {batch_x.shape}")
                # print(f"batch_t shape: {batch_t.shape}")

                # 勾配を求め、パラメータを更新
                loss = model.forward(batch_x, batch_t)
                model.backward()
                params, grads = remove_duplicate(model.params, model.grads)  # 共有された重みを1つに集約
                if max_grad is not None:
                    clip_grads(grads, max_grad)
                optimizer.update(params, grads)
                total_loss += loss
                loss_count += 1

                # パープレキシティの評価
                if (eval_interval is not None) and (iters % eval_interval) == 0:
                    ppl = np.exp(total_loss / loss_count)
                    elapsed_time = time.time() - start_time
                    print('| epoch %d |  iter %d / %d | time %d[s] | perplexity %.2f'
                          % (self.current_epoch + 1, iters + 1, max_iters, elapsed_time, ppl))
                    self.loss_list.append(float(ppl))
                    total_loss, loss_count = 0, 0
            self.current_epoch += 1


    def plot(self, ylim=None):
        x = numpy.arange(len(self.loss_list))
        if ylim is not None:
            plt.ylim(*ylim)

        plt.plot(x, self.loss_list, label='train')
        plt.xlabel(f'iterations (x + {str(self.eval_interval)} + )')
        plt.ylabel('perplexity')
        plt.show()

    

def remove_duplicate(params, grads):
    """
    パラメータ配列の重複する重みを一つに集約し、
    その重みに対応する勾配を加算する
    Args:
        params ([type]): [description]
        grads ([type]): [description]
    """
    params, grads = params[:], grads[:] # copy list

    while True:
        find_flg = False
        L = len(params)

        # [0,1,2,3] -> (0,1)(0,2)(0,3)(1,2)(1,3)(2,3) 4_C_2 = 6通り
        for i in range(0, L - 1):
            for j in range(i + 1, L):

                # 重みを共有する場合
                if params[i] is params[j]:
                    grads[i] += grads[j] # 勾配の加算
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)
                
                # 転地行列として重みを共有する場合(weight tying)
                elif params[i].ndim == 2 and \
                     params[j].ndim == 2 and \
                     params[i].T.shape == params[j].shape and \
                     np.all(params[i].T == params[j]):
                    grads[i] += grads[j].t
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)
                
                if find_flg:
                    break
            if find_flg:
                break
        
        if not find_flg:
            break

    return params, grads



