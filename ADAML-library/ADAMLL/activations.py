import jax.numpy as np


def eye(x):
    return x

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.where(x > 0, x, 0)

def tanh(x):
    return np.tanh(x)

def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def leaky_relu(x, alpha=0.01):
    return np.maximum(alpha*x, x)

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha*(np.exp(x)-1))

def swish(x, beta=1.0):
    return x / (1 + np.exp(-beta*x))

def mish(x):
    return x * np.tanh(np.log(1 + np.exp(x)))

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * np.power(x, 3))))

def softplus(x):
    return np.log(1 + np.exp(x))

def softsign(x):
    return x / (1 + np.abs(x))
