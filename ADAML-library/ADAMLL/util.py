import jax.numpy as np
import numpy as onp
import jax


@jax.jit
def MSE(y_data:np.ndarray ,y_model:np.ndarray  ) -> float:
    """
    Calculates the mean squared error
    """
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n




@jax.jit
def R2(y_data:np.ndarray ,y_model:np.ndarray ) -> float:
    """
    Calculates the R2 score
    """
    return 1 - np.sum((y_data-y_model)**2)/np.sum((y_data-np.mean(y_model))**2)


@jax.jit
def CE(y, t):
    """
    Cross entropy loss function
    """
    return -np.mean(t * np.log(y) + (1 - t) * np.log(1 - y))


@jax.jit
def accuracy(y, t):
    """
    Calculates the accuracy of a classification model
    """
    return np.mean(y == t) 





def one_hot_encode(y, n_classes):
    """
    One hot encode a vector
    """
    return np.eye(n_classes)[y]




def confusion_matrix(y, t):
    """ binary confusion matrix """
    tp = np.sum(np.logical_and(y == 1, t == 1))
    tn = np.sum(np.logical_and(y == 0, t == 0))
    fp = np.sum(np.logical_and(y == 1, t == 0))
    fn = np.sum(np.logical_and(y == 0, t == 1))
    return np.array([[tp, fp], [fn, tn]])







def progress_bar(current, total, bar_length=20):
    fraction = current / total

    arrow = int(fraction * bar_length - 1) * '-' + '>'
    padding = int(bar_length - len(arrow)) * ' '

    return f'Progress: [{arrow}{padding}] {int(fraction*100)}%'



def print_message(message):
    print(f"\r{message: <70}", end='')




def generate_perlin_noise(n, m, seed=None, scale=0.025):
    if seed is None:
        key = jax.random.PRNGKey(1)
    else:
        key = jax.random.PRNGKey(seed)

    def fade(t):
        return t * t * t * (t * (t * 6 - 15) + 10)

    def lerp(t, a, b):
        return a + t * (b - a)

    def gradient(h, x, y):
        gradients = np.array([[1, 1], [-1, 1], [1, -1], [-1, -1]])
        i = h % 4
        return gradients[i].dot(np.array([x, y]))

    def perlin(x, y):
        X = int(x) & 255
        Y = int(y) & 255
        x -= int(x)
        y -= int(y)
        u = fade(x)
        v = fade(y)
        A = p[X] + Y
        B = p[X + 1] + Y
        return lerp(v, lerp(u, gradient(p[A], x, y), gradient(p[B], x - 1, y)), lerp(u, gradient(p[A + 1], x, y - 1), gradient(p[B + 1], x - 1, y - 1)))

    p = jax.random.permutation(key, np.arange(256, dtype=np.uint32),independent=True) 
    p = np.hstack([p, p]) 

    perlin_noise = np.zeros((n, m))

    for i in range(n):
        for j in range(m):
            perlin_noise = perlin_noise.at[i,j].set((perlin(i * scale, j * scale)))

    return perlin_noise
