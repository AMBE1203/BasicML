from __future__ import print_function
import numpy as np

# hàm số f(x) = x^2 + 10sin(x)

def grad(x):
    return 2*x + 10*np.cos(x)

def cost(x):
    return x ** 2 + 10*np.sin(x)

def GD_basic(x0, eta):
    x = [x0]
    for i in range(100):
        x_new = x[-1] - eta*grad(x[-1])
        if abs(grad(x_new)) < 1e-3:
            break
        x.append(x_new)
    return (x, i)

(x1, it) = GD_basic(5, .1)
print('Solution x1 = %f, cost = %f, after %d iteration' %(x1[-1], cost(x1[-1]),it))

def GD_momentum(grad, theta_init, eta, gamma):
    theta = [theta_init]
    v_old = np.zeros_like(theta_init)
    for i in range(100):
        v_new = gamma*v_old + eta*grad(theta[-1])
        theta_new = theta[-1] - v_new
        if np.linalg.norm(grad(theta_new))/ np.array(theta_init).size < 1e-3:
            break
        theta.append(theta_new)
        v_old = v_new
    return (theta, i)

(x1, it) = GD_momentum(grad, 5, .1, .9)
print('Solution with momentum x1 = %f, cost = %f, after %d iteration' %(x1[-1], cost(x1[-1]),it))

def GD_NAG(grad, theta_init, eta, gamma):
    # suppose we want to store history of theta
    theta = [theta_init]
    v_old = np.zeros_like(theta_init)
    for i in range(100):
        v_new = gamma*v_old + eta*grad(theta[-1] - gamma*v_old)
        theta_new = theta[-1] - v_new
        if np.linalg.norm(grad(theta_new)) / np.array(theta_init).size < 1e-3:
            break
        theta.append(theta_new)
        v_old = v_new
    return (theta, i)

    (x1, it) = GD_NAG(grad, 5, .1, .9)
print('Solution with NAG x1 = %f, cost = %f, after %d iteration' %(x1[-1], cost(x1[-1]),it))