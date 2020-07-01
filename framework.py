#!/usr/bin/env python3

## You will have to implement a very primitive machine learning framework. The skeleton is given, you have to fill the
## blocks marked by "## Implement". The goal is to give you a little insight how machine learning frameworks work,
## while practicing the most important details from the class.
##
## You have to implement the following modules (or call them layers?):
##      * Activation functions: Tanh, Sigmoid
##      * Layers: Linear, Sequential
##      * Losses: MSE (mean-squared error)
##
## Linear layer is also called linear perceptron, fully connected layer, etc. The bias term is not included in the
## network weight matrix W, because of performance reasons (concatenating the 1 to the input is slow). This is the case
## for the real-world ML frameworks too.
##
## Sequential layer receives a list of layers for the constructor argument, and it calls them in order on forward and in
## reverse order on backward. It is just a syntactic sugar that makes the code much nicer. You can call a Tanh after a
## Sigmoid layer by calling net = Sequential([Sigmoid(), Tanh()]), output = net.forward(your data).
##
## All the modules have a forward() and a backward() function. forward() receives one argument (except for the loss) and
## returns the output of that layer. The backward() receives the error flowing back on the output of the layer, and
## should return 2 things (a tuple of 2):
##      1: a dict of gradients, where the keys are the same as in the .var member of the layer and the values are
##         numpy arrays representing the gradient for that variable.
##      2: a numpy array representing the gradient for the input (that was passed to the layer in the forward pass).
## The backward() does not receive the input, although it might be needed for the gradient calculation.
## You should save them in the forward pass for later use in the backward pass. You don't have to worry about
## the most of this, as it is already implemented in the skeleton. There are 2 imporant takeaways: you have to calculate
## the gradient both of your variables and the layer input in the backward pass, and if you need the layer input, then
## you need to save it in a class variable.
##
## You will also have to implement the function train_one_step(), which does one step of weight update based on the
## training data and the learning rate.
##
## After implementing the gradinet check, you can be almost sure that your backward functions are correct. In order to
## do this, you will have to fill in the analytic and numerical gradient computation part of the gradient_check()
## function. What this does is it iterates over all the elements of all variables, nudges it a bit in both directions,
## and recalculates the network output. Based on that, you can calculate what the gradient should be if we assume that
##  the forward pass is correct.
##
## Finally you would have to complete the create_network() function, which should return a Sequential neural network of
## 3 layers: a tanh input layer with 2 inputs and 50 outputs, a tanh hidden layer with 30 outputs and finally a sigmoid
## output layer with 1 output.
##
## At the end of the training your cost should be around 0.008. Don't be afraid if it differs a bit, but a significantly
##  higher value may indicate a problem.
##
## At many points in the code there are asserts which check the shapes of the gradients. Remember: the gradient for a
## variable must have the same shape as the variable itself. Imagine the variables and the network inputs/outputs as a
## cable with a given number of wires: no matter in which direction you pass the data, the number of wires is the same.
##
## Take a special care about implementing gradient checking, as it can diagnose the rest of the assignment. Use symetric
## numerical gradient calculation. If your gradient checking passes and your error is around 0.008, your solution is
## probably correct. It's worth trying to intentionally corrupt the backward of some module (for example Tanh) by a
## tiny bit (~0.01) and see if the gradcheck fails. If not, your gradcheck might be wrong. Just don't forget to change
## it back to the correct value before submitting.
##
## Please do your calculations in a vectorized way, otherwise it will be painfully slow. You have to use for loop only
## twice.
##
## This script needs numpy and matplotlib to run. You can install with pip3 install numpy and pip3 install matplotlib.
##
## Good luck, I hope you'll enjoy it :)!

import numpy as np


class Tanh:
    def forward(self, x):
        result = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

        self.saved_variables = {
            "result": result
        }
        return result

    def backward(self, error):
        tanh_x = self.saved_variables["result"]

        ## Implement
        #error e' gia la derivata propagata, quindi devo solo calcolare la derivata di tanh e moltiplicarla per error

        d_x = np.multiply((1 - np.power(tanh_x, 2)), error)

        ## End
        assert d_x.shape == tanh_x.shape, "Input: grad shape differs: %s %s" % (d_x.shape, tanh_x.shape)

        self.saved_variables = None
        return None, d_x


class Sigmoid:
    def forward(self, x):
        ## Implement

        result = 1 / (1 + np.exp(-x))

        ## End
        self.saved_variables = {
            "result": result
        }
        return result

    def backward(self, error):
        sigmoid_x = self.saved_variables["result"]

        ## Implement
        
        d_x = np.multiply(sigmoid_x * (1 - sigmoid_x), error)

        ## End
        assert d_x.shape == sigmoid_x.shape, "Input: grad shape differs: %s %s" % (d_x.shape, sigmoid_x.shape)

        self.saved_variables = None
        return None, d_x


class Linear:
    def __init__(self, input_size, output_size):
        self.var = {
            "W": np.random.normal(0, np.sqrt(2 / (input_size + output_size)), (input_size, output_size)),
            "b": np.zeros((output_size), dtype=np.float32)
        }

    def forward(self, inputs):
        x = inputs
        W = self.var['W']
        b = self.var['b']

        ## Implement
        ## Save your variables needed in backward pass to self.saved_variables.

        y = np.matmul(x, W) + b

        ## End
        self.saved_variables = {
            "inputs": x,
            "weights": W
        }
        return y

    def backward(self, error):
        ## Implement
        x = self.saved_variables["inputs"]
        dW = np.matmul(np.transpose(x), error)
        db = np.matmul(np.ones(error.shape[0]), error)

        #Questa e' la derivata del prodotto matriciale rispetto a x? Si
        d_inputs = np.matmul(error, np.transpose(self.saved_variables["weights"]))

        ## End
        assert d_inputs.shape == x.shape, "Input: grad shape differs: %s %s" % (d_inputs.shape, x.shape)
        assert dW.shape == self.var["W"].shape, "W: grad shape differs: %s %s" % (dW.shape, self.var["W"].shape)
        assert db.shape == self.var["b"].shape, "b: grad shape differs: %s %s" % (db.shape, self.var["b"].shape)

        self.saved_variables = None
        updates = {"W": dW,
                   "b": db}
        return updates, d_inputs


class Sequential:
    def __init__(self, list_of_modules):
        self.modules = list_of_modules

    class RefDict(dict):
        #k = Mod_0.W, d = dizionario(var), key = W, e setta un elemento con chiave k e come valore una coppia dizionario
        # e nome della var
        def add(self, k, d, key):
            super().__setitem__(k, (d, key))
        #Questa viene chiamata quando uso le []
        def __setitem__(self, k, v):
            assert k in self, "Trying to set a non-existing variable %s" % k
            ref = super().__getitem__(k)
            ref[0][ref[1]] = v

        def __getitem__(self, k):
            ref = super().__getitem__(k)
            return ref[0][ref[1]]

        def items(self):
            for k in self.keys():
                yield k, self[k]

    @property
    def var(self):
        res = Sequential.RefDict()
        for i, m in enumerate(self.modules):
            #Enumerate ritorna delle coppie i,m con i che contiene un indice crescente m contiene il modulo
            if not hasattr(m, 'var'):
                continue

            for k in m.var.keys():
                res.add("mod_%d.%s" % (i, k), m.var, k)
        return res

    def update_variable_grads(self, all_grads, module_index, child_grads):
        #Se e' None vuol dire che non era Linear quindi non ho gradienti da aggiornare
        if child_grads is None:
            return all_grads
        #Se e' la prima volta lo creo
        if all_grads is None:
            all_grads = {}

        for name, value in child_grads.items():
            all_grads["mod_%d.%s" % (module_index, name)] = value

        return all_grads

    def forward(self, input):
        ## Implement
        result = input
        for module in self.modules:
            result = module.forward(result)

        return result
        ## End

    def backward(self, error):
        variable_grads = None

        for module_index in reversed(range(len(self.modules))):
            module = self.modules[module_index]

            ## Implement
            tmp = module.backward(error)
            # module_variable_grad e' null nel caso non linear
            module_variable_grad = tmp[0]
            module_input_grad = tmp[1]

            ## End
            ## L'errore da propagare al layer precedente e' la derivata rispetto all'input per l'errore
            error = module_input_grad
            variable_grads = self.update_variable_grads(variable_grads, module_index, module_variable_grad)

        return variable_grads, module_input_grad


class MSE:
    def forward(self, prediction, target):
        Y = prediction
        T = target
        n = prediction.size

        ## Implement
        ## Don't forget to save your variables needed for backward to self.saved_variables..

        meanError = np.multiply(np.sum(np.power(T - Y, 2)), 0.5) * (1 / n)
        self.saved_variables = {
            "prediction": Y,
            "target": T,
            "n": n
        }
        ## End
        return meanError

    def backward(self):
        #Considering as error function (target - net)ˆ2 (Alippi)
        ## Implement
        y = self.saved_variables["prediction"]
        d_prediction = (y - self.saved_variables["target"]) * (1/self.saved_variables["n"])

        ## End
        assert d_prediction.shape == y.shape, "Error shape doesn't match prediction: %d %d" % \
                                              (d_prediction.shape, y.shape)

        self.saved_variables = None
        return d_prediction

def train_one_step(model, loss, learning_rate, inputs, targets):
    ## Implementß
    error = loss.forward(model.forward(inputs), targets)
    variable_gradients, _ = model.backward(loss.backward())
    #delta rule
    for k, v in enumerate(model.modules):
        if not hasattr(v, 'var'):
            continue
        v.var['W'] = v.var['W'] - (learning_rate * variable_gradients["mod_%d.%s" % (k, 'W')])
        v.var['b'] = v.var['b'] - (learning_rate * variable_gradients["mod_%d.%s" % (k, 'b')])

    ## End
    return error

#a tanh input layer with 2 inputs and 50 outputs, a tanh hidden layer with 30 outputs and finally a sigmoid
## output layer with 1 output
def create_network():
    ## Implement
    
    network = Sequential([Linear(2, 50), Tanh(), Linear(50,30), Tanh(), Linear(30,1), Sigmoid()])

    ## End
    return network


def gradient_check():
    X, T = twospirals(n_points=10)
    NN = create_network()
    eps = 0.0001

    loss = MSE()
    loss.forward(NN.forward(X), T)
    variable_gradients, _ = NN.backward(loss.backward())

    all_succeeded = True

    # Check all variables. Variables will be flattened (reshape(-1)), in order to be able to generate a single index.
    for key, value in NN.var.items():
        #Quando uso le [] chiamo la mia get che mi ritorna var[W o b]
        variable = NN.var[key].reshape(-1)
        variable_gradient = variable_gradients[key].reshape(-1)
        success = True

        if NN.var[key].shape != variable_gradients[key].shape:
            print("[FAIL]: %s: Shape differs: %s %s" % (key, NN.var[key].shape, variable_gradients[key].shape))
            success = False
            break

        # Check all elements in the variable
        for index in range(variable.shape[0]):
            var_backup = variable[index]

            ## Implement

            analytic_grad = variable_gradient[index]

            variable[index] = var_backup + eps
            forward_positive_nudge = loss.forward(NN.forward(X), T)
            variable[index]= var_backup - eps
            forward_negative_nudge = loss.forward(NN.forward(X), T)

            numeric_grad = (forward_positive_nudge - forward_negative_nudge) / (2*eps)
            ## End

            variable[index] = var_backup
            if abs(numeric_grad - analytic_grad) > 0.00001:
                print("[FAIL]: %s: Grad differs: numerical: %f, analytical %f" % (key, numeric_grad, analytic_grad))
                success = False
                break

        if success:
            print("[OK]: %s" % key)

        all_succeeded = all_succeeded and success

    return all_succeeded


###############################################################################################################
# Nothing to do past this line.
###############################################################################################################

if __name__ == "__main__":
    import matplotlib
    import matplotlib.pyplot as plt

    np.random.seed(0xDEADBEEF)

    plt.ion()


    def twospirals(n_points=120, noise=1.6, twist=420):
        """
         Returns a two spirals dataset.
        """
        np.random.seed(0)
        n = np.sqrt(np.random.rand(n_points, 1)) * twist * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + np.random.rand(n_points, 1) * noise
        d1y = np.sin(n) * n + np.random.rand(n_points, 1) * noise
        X, T = (np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))),
                np.hstack((np.zeros(n_points), np.ones(n_points))))
        T = np.reshape(T, (T.shape[0], 1))
        return X, T


    fig, ax = plt.subplots()


    def plot_data(X, T):
        ax.scatter(X[:, 0], X[:, 1], s=40, c=T.squeeze(), cmap=plt.cm.Spectral)


    def plot_boundary(model, X, targets, threshold=0.0):
        ax.clear()
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
        X_grid = np.c_[xx.ravel(), yy.ravel()]
        y = model.forward(X_grid)
        ax.contourf(xx, yy, y.reshape(*xx.shape) < threshold, alpha=0.5)
        plot_data(X, targets)
        ax.set_ylim([y_min, y_max])
        ax.set_xlim([x_min, x_max])
        plt.show()
        plt.draw()
        plt.pause(0.001)


    def main():        
        print("Checking the network")
        if not gradient_check():
            print("Failed. Not training, because your gradients are not good.")
            return
        print("Done. Training...")
        X, T = twospirals(n_points=200, noise=1.6, twist=600)
        NN = create_network()
        loss = MSE()

        learning_rate = 0.1

        for i in range(20000):
            curr_error = train_one_step(NN, loss, learning_rate, X, T)
            if i % 200 == 0:
                print("step: ", i, " cost: ", curr_error)
                plot_boundary(NN, X, T, 0.5)

        plot_boundary(NN, X, T, 0.5)
        print("Done. Close window to quit.")
        plt.ioff()
        plt.show()



    main()