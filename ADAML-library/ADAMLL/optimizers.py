import jax
import jax.numpy as np


############################################################################################################
# SGD
############################################################################################################

def create_update_sgd(eta, gamma):
    def sgd_update(params, grads, state ):
        momentum = state
        new_momentum = jax.tree_map(lambda m,g: gamma * m + eta * g, momentum, grads )
        new_params = jax.tree_map(lambda p, m: p - m, params, new_momentum)
        return new_params, new_momentum
    return sgd_update



def init_SGD_state(params):
    return jax.tree_map(lambda p: np.zeros_like(p), params)



############################################################################################################
# ADAGRAD
############################################################################################################

# with momentum
def create_update_adagrad(eta=0.001,  epsilon=1e-8):
    def adagrad_update(params, grads, state ):
        v = state
        v = jax.tree_map(lambda v, g: v + g ** 2, v, grads)
        new_params = jax.tree_map(lambda p, v,g : p - eta * g / (np.sqrt(v) + epsilon), params, v, grads)
        return new_params, v
    return adagrad_update



def init_adagrad_state(params):
    return jax.tree_map(lambda p: np.zeros_like(p), params)  # Initialize second moment






############################################################################################################
# ADAM
############################################################################################################

def create_update_adam(eta=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    def adam_update(params, grads, state ):
        m, v, t = state
        m = jax.tree_map(lambda m_i, g_i: beta1 * m_i + (1 - beta1) * g_i, m, grads)
        v = jax.tree_map(lambda v_i, g_i: beta2 * v_i + (1 - beta2) * (g_i ** 2), v, grads)
        m_hat = jax.tree_map(lambda m_i: m_i / (1 - beta1 ** (t + 1)), m)
        v_hat = jax.tree_map(lambda v_i: v_i / (1 - beta2 ** (t + 1)), v)
        new_params = jax.tree_map(lambda p, m_h, v_h: p - (eta * m_h / (np.sqrt(v_h) + epsilon)), params, m_hat, v_hat)
        new_t = t + 1
        new_state = (m, v, new_t)
        return new_params, new_state
    return adam_update



def init_adam_state(params):
    ms = jax.tree_map(lambda p: np.zeros_like(p), params)  # Initialize first moment
    vs = jax.tree_map(lambda p: np.zeros_like(p), params)  # Initialize second moment
    ts = 0 
    return ms, vs, ts




############################################################################################################
# RMSprop
############################################################################################################

def create_update_rmsprop(eta=0.001, gamma=0.9, epsilon=1e-8):
    def rmsprop_update(params, grads, state ):
        v = state
        v = jax.tree_map(lambda v,g : gamma * v + (1 - gamma) * (g ** 2), v, grads)
        new_param = jax.tree_map(lambda p,v,g: p - eta * g / (np.sqrt(v) + epsilon), params, v, grads)
        return new_param, v
    return rmsprop_update


def init_rmsprop_state(params):
    return jax.tree_map(lambda p: np.zeros_like(p), params)  # Initialize second moment


