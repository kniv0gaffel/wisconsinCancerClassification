import jax.numpy as np
import jax
from .util import  MSE, CE, print_message
from .activations import sigmoid, eye,  softmax, relu, tanh, leaky_relu, elu, swish, mish, gelu, softplus, softsign
from . import optimizers





############################################################################################################
# INITIALIZATION
############################################################################################################
# sets the weights and biases of the network to random values
def init_network_params(architecture, key):
    layer_sizes = [l[0] for l in architecture]
    keys = jax.random.split(key, len(layer_sizes) - 1)
    return [{'w': jax.random.normal(k, (in_size, out_size)) * np.sqrt(2 / in_size),
             'b': np.zeros(out_size)}
            for k, in_size, out_size in zip(keys, layer_sizes[:-1], layer_sizes[1:])]





############################################################################################################
# NETWORK CLASS
############################################################################################################

class Model():
    """
    A class for a neural network
    """

    def __init__(self ,architecture=[ [2, sigmoid], [1, eye]  ],
                eta=0.1, epochs=100, tol=0.001, optimizer='sgd', alpha=0,
                 gamma=0, epsilon=0.0001,  beta1=0.9, beta2=0.999, backwards=None, loss=MSE, metric=MSE ):

        self.metric = metric
        self.eta = eta
        self.epochs = epochs
        self.tol = tol
        self.optimizer = optimizer
        self.gamma = gamma
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
        self.architecture = architecture
        self.architecture.insert(0, []) 

        if backwards == None:
            self.backwards = build_backwards( alpha=alpha, loss=loss )
            self.forward = build_forward(self.architecture, auto_diff=True )
        else:
            self.backwards = backwards(alpha=alpha)
            self.forward = build_forward(self.architecture, auto_diff=False )




    def init_optimizer(self, params):
        match self.optimizer:
            case 'sgd':
                optimizer = optimizers.create_update_sgd(self.eta, self.gamma)
                optimizerState = optimizers.init_SGD_state(params)
                return optimizer, optimizerState
            case 'adagrad':
                optimizer = optimizers.create_update_adagrad(self.eta, self.epsilon )
                optimizerState = optimizers.init_adagrad_state(params)
                return optimizer, optimizerState
            case 'rmsprop':
                optimizer = optimizers.create_update_rmsprop(self.eta, self.epsilon, self.gamma)
                optimizerState = optimizers.init_rmsprop_state(params)
                return optimizer, optimizerState
            case 'adam':
                optimizer = optimizers.create_update_adam(self.eta, self.beta1, self.beta2, self.epsilon)
                optimizerState = optimizers.init_adam_state(params)
                return optimizer, optimizerState
            case _:
                raise ValueError(f"Unknown optimizer {self.optimizer}")



    def fit(self, X, t, X_val=None, t_val=None, batch_size=None ):
        """
        X: training data
        t: training labels
        X_val: validation data
        t_val: validation labels
        batch_size: size of the batches

        returns: loss, params

        fits the network to the data
        """
        N,n = X.shape
        self.architecture[0] = [n]
        if batch_size is None:
            batch_size = N

        key = jax.random.PRNGKey(1234)
        params = init_network_params(self.architecture, key)
        update_params, opt_state = self.init_optimizer(params) 
        batches = int(N/batch_size)
        loss = np.zeros(self.epochs)

        @jax.jit # one step of gradient descent jitted to make it zoom
        def step(params, opt_state, X, t):
            activations, grads = self.forward(params, X)
            grads = self.backwards(params, t, activations, grads )
            # grads = clip_gradients_by_norm(grads, 1) # gradient clipping
            params, opt_state = update_params(params, grads, opt_state)
            return params, opt_state

        if X_val is None:

            for e in range(self.epochs):
                for _ in range(batches):

                    key, subkey = jax.random.split(key)
                    random_index = batch_size * jax.random.randint(subkey, minval=0, maxval=batches, shape=())
                    X_batch = X[random_index:random_index+batch_size]
                    t_batch = t[random_index:random_index+batch_size]

                    params, opt_state= step(params, opt_state, X_batch, t_batch)


        else:
            for e in range(self.epochs):
                for _ in range(batches):

                    key, subkey = jax.random.split(key)
                    random_index = batch_size * jax.random.randint(subkey, minval=0, maxval=batches, shape=())
                    X_batch = X[random_index:random_index+batch_size]
                    t_batch = t[random_index:random_index+batch_size]

                    params, opt_state= step(params, opt_state, X_batch, t_batch)
                    current_loss = self.metric(self.forward(params, X_val)[0][-1], t_val, X_val) # this has changed when solving the wave equation

                    loss = loss.at[e].set(current_loss)

                    # Early stopping condition
                    if e > 10 and np.abs(loss[e-10] - loss[e]) < self.tol:
                        loss = loss.at[e+1:].set(loss[e]) 
                        break


        print_message(f"Training stopped after {e} epochs")
        self.params = params
        return loss , params
    



    def predict(self, X):
        """
        X: data to predict on

        returns: predictions
        """
        return self.forward(self.params, X)[0][-1]




    def classify(self, X):
        """
        X: data to predict on

        returns: binary predictions
        """
        return np.round(self.predict(X))

   
















############################################################################################################
# BACKWARD PROPAGATION
############################################################################################################


# COLLECTION OF BACKWARD FUNCTIONS, USING MANUAL DIFFERENTIATION
# IF THESE ARE USED, THE FORWARD PROPAGATION FUNCTION MUST BE SET TO AUTO_DIFF = FALSE


def backwards_one_hidden(alpha=0.0):
    def backwards(params, t, activations, _ ):
        """Cross entropy with sigmoid - sigmoid"""
        X, a0, a1 = activations
        output_error = (a1 - t) / (X.shape[0])
        hidden_error =  output_error @ params[1]['w'].T * a0 * (1 - a0)

        gw0 = X.T @ hidden_error   + 2 * alpha * params[0]['w']
        gb0 = np.sum(hidden_error, axis=0)
        gw1 = a0.T @ output_error  + 2 * alpha * params[1]['w']
        gb1 = np.sum(output_error, axis=0)
        return [{'w': gw0, 'b': gb0}, {'w': gw1, 'b': gb1}]
    return backwards





def backwards_no_hidden(alpha=0.0 ):
    def backwards(params, t, activations, _ ):
        """Cross entropy with sigmoid"""
        X, y = activations
        wgrad = (2/X.shape[0]) * np.dot(X.T , (y - t)) + 2 * alpha * params[0]['w']
        bgrad = 2/X.shape[0] * np.sum(y - t)
        return [{'w': wgrad, 'b': bgrad}]
    return backwards




# FULLY AUTOMATIC BACKWARD PROPAGATION, ADAPTS TO ANY* ARCHITECTURE

def build_backwards( alpha, loss=CE):

    @jax.jit
    def backwards(params, t, activations, grads ):
        a0, a1 = activations[-2:] # last two activations
        x = activations[0]                                  # this has changed when solving the wave equation
        g1 = grads[-1]
        loss_deriv = jax.grad(loss)(a1, t, x)               # this has changed when solving the wave equation
        output_error = loss_deriv * g1
        gb = np.sum(output_error, axis=0)
        gw = a0.T @ output_error + 2 * alpha * params[-1]['w']
        param_gradients = [{'b': gb, 'w': gw}]
        for i in range(len(params) - 1, 0 , -1): 
            g = grads[i-1]
            output_error = output_error @ params[i]['w'].T * g
            a0 = activations[i-1]
            gw = a0.T @ output_error + 2 * alpha * params[i-1]['w']
            gb = np.sum(output_error, axis=0)
            param_gradients.insert(0,{'b': gb, 'w': gw})
        return param_gradients
    return backwards



def clip_gradients_by_norm(gradients, max_norm):
    total_norm = np.sqrt(sum(np.sum(np.square(g)) for g in jax.tree_leaves(gradients)))
    clip_coeff = max_norm / (total_norm + 1e-6)
    clipped_gradients = jax.tree_map(lambda g: np.where(clip_coeff < 1, clip_coeff * g, g), gradients)
    return clipped_gradients


############################################################################################################
# FORWARD PROPAGATION
############################################################################################################


def build_forward(architecture, auto_diff=True ):
    architecture = architecture[1:]
    acitvation_functions = [l[1] for l in architecture]

    @jax.jit
    def forward(network, inputs):
        activations = [inputs]
        for i in range(len(network)): 
            z = np.dot(activations[i], network[i]['w']) + network[i]['b']
            activations.append(acitvation_functions[i](z))
        return activations, None


    @jax.jit
    def forward_auto(network, inputs):
        activations = [inputs]
        grads = []
        for i in range(len(network)): 
            value_and_grad = jax.vmap(jax.value_and_grad(acitvation_functions[i]))
            z = np.dot(activations[i], network[i]['w']) + network[i]['b']
            a , g = value_and_grad(z.ravel())
            activations.append(a.reshape(z.shape))
            grads.append(g.reshape(z.shape))
        return activations, grads 


    if auto_diff:
        return forward_auto
    else:
        return forward


