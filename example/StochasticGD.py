from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(12)

X = np.random.rand(10, 1)
y = 4 + 3*X + .2*np.random.randn(10, 1)
ones = np.ones((X.shape[0], 1))
Xbar = np.concatenate((ones, X), axis = 1)

# single point gradient
def sgrad(w, i, rd_id):
    true_i = rd_id[i]
    xi = Xbar[true_i, :]
    yi = y[true_i]
    a = np.dot(xi, w) - yi
    return (xi * a).reshape(2, 1)

def SDG(w_init, sgrad, eta):
    w = [w_init]
    w_last_check = w_init
    iter_check_w = 10
    N = X.shape[0]
    count = 0
    for it in range(10):
        # shuffle data (Xáo trộn dữ liệu)
        rd_id = np.random.permutation(N)
        for i in range(N):
            count += 1
            g = sgrad(w[-1], i, rd_id)
            w_new = w[-1] - eta*g
            w.append(w_new)
            if count%iter_check_w == 0:
                w_this_check = w_new
                if np.linalg.norm(w_this_check - w_last_check) / len(w_init) < 1e-3:
                    return w
                w_last_check = w_this_check
    return (w, it)


w_init = np.array([[2], [1]])
(w, it) = SDG(w_init, sgrad, .1)
print('Sol found by SGD: w = ', w[-1].T, '\nafter %d iterations.' %(it+1))