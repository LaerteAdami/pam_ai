import numpy as np


class AdamOptimizer:
    """
    Adam optimizer to be used in the training of a neural network

    """

    def __init__(self):

        self.learning_rate = 0.001
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-07

    def update_weigths(self, dweigths, moment1, moment2):
        """
        Updates weights by means of moments computed by the optimizer

        """

        moment1 = self.beta1 * moment1 + (1 - self.beta1) * dweigths
        moment2 = self.beta2 * moment2 + (1 - self.beta2) * dweigths * dweigths

        return (
            moment1,
            moment2,
            self.learning_rate * moment1 / (np.sqrt(moment2) + self.epsilon),
        )

    def update_bias(self, dbias, moment1, moment2):
        """
        Updates biases by means of moments computed by the optimizer

        """

        moment1 = self.beta1 * moment1 + (1 - self.beta1) * dbias
        moment2 = self.beta2 * moment2 + (1 - self.beta2) * dbias * dbias

        return (
            moment1,
            moment2,
            self.learning_rate * moment1 / (np.sqrt(moment2) + self.epsilon),
        )
