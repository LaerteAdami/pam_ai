import numpy as np
from scipy.stats import truncnorm

# Function to initialise the weigths and the biases
def truncated_normal(mean=0, sd=1, low=0, upp=5):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

class NeuralNetwork():
    """
    Fully connected neural network

    """

    def __init__(self, learning_rate,max_epochs,type_loss=["mse","cross_entropy"], optimizer = None):
        self.lr = learning_rate
        self.max_epochs = max_epochs
        self.type_loss = type_loss
        self.layers = []
        self.opt = optimizer
        
    def add(self, Layer):
        """
        Adds layers to the network (from input to output + hidden layers)
        
        """    
        self.layers.append(Layer) # add a layer
        
    def build_network(self):
        """
        Initialises weights matrices in the network
    
        """
        
        # Initiate the optimizer
        if self.opt:
            self.opt.learning_rate = self.lr
        
        for i in range(1,len(self.layers)):
            
            rad = 0.5 
            tn = truncated_normal(mean=1, sd=1, low=-rad, upp=rad)  # type: ignore
            self.layers[i].w = tn.rvs((self.layers[i-1].n,self.layers[i].n))  # initialise weights matrices for layers
            self.layers[i].b = np.zeros((1,self.layers[i].n)) # initialise biases for layers
            
            # Set optimizer
            self.layers[i].optimizer = self.opt
            
            # For Adam Optimizer
            self.layers[i].w_moment1 = np.zeros_like(self.layers[i].w)
            self.layers[i].w_moment2 = np.zeros_like(self.layers[i].w)
            self.layers[i].b_moment1 = np.zeros_like(self.layers[i].b)
            self.layers[i].b_moment2 = np.zeros_like(self.layers[i].b)

        
    def loss(self,y_pred, y_test):
        """
        Definition of the loss function: mse for regression and cross_entropy for classification
        
        """
        
        if self.type_loss == "mse":
        
            dif = y_pred-y_test
            
            return (1/2)*np.dot(dif.T,dif)  # type: ignore 
       
        elif self.type_loss == "cross_entropy":
        
            return -np.sum(y_test * np.log(y_pred))  # type: ignore 

    def d_loss(self,y_pred, y_test):
        """
        Definition of the derivative of loss function: mse for regression and cross_entropy for classification
        
        """
        
        if self.type_loss == "mse":

            return y_pred-y_test
        
        elif self.type_loss == "cross_entropy":
                 
            return y_pred-y_test # here y_pred is the probability given by softmax layer
       
    def fit_batch(self, X, y, toll):
        """
        Trains the neural network with batch gradient descent technique
        
        """
        
        L_history = [] # stores values of the loss 
        
        for epoch in range(self.max_epochs):
            
            # Apply dropout
            active_indices = self.dropout_matrices()
            input_indices = active_indices[-1]
            
            io_layer = X[:,input_indices] # input/output of each layer BIAS
            
            # Compute the forward pass of the network --> y*
            for i in range(1,len(self.layers)):
                
                # compute output
                layer = self.layers[i]
                
                io_layer = layer.forward_pass(io_layer)     
            
            # Compute the loss function and its derivative
            L = self.loss(io_layer, y)
            L_history.append(L)
            dL = self.d_loss(io_layer, y) # this will be the first error to be transferred backwards
                
            io_error = dL # initialise the first gradient with the derivative of the loss

            for i in range(len(self.layers)-1,0,-1): # from last layer to the first layer
 
                # set layer 
                layer = self.layers[i]
            
                # update weights and compute error to pass at the previous layer
                io_error = layer.backward_pass(io_error,self.lr)
                
            # Restore matrices of the dropout    
            self.restore_matrices()
            
            # Stopping criterion
            if epoch > 3:
                if abs(L_history[-1] - L) <= toll and abs(L_history[-2] - L) <= toll and abs(L_history[-3] - L) <= toll:
                    print("Stop!")
                    break

        return L_history        
        
    def predict(self, X):
        """
        Predict values after training of the model
        
        """
        
        io_layer = X
        
        for i in range(1,len(self.layers)):
                
            # compute output
            layer = self.layers[i]
            io_layer = layer.forward_pass(io_layer) 
            
        return io_layer  
    
    
    def dropout_matrices(self):
        """
        Applies the dropout to the weight matrices in the neural network
        
        """
                                            
        indices = range(self.layers[-1].n) # initialiase indices - no dropout in output layer
        
        indices_active_nodes = []
        
        for i in range(len(self.layers) - 1,0, - 1): # from last layer to the first layer

            # set layer 
            layer = self.layers[i]
            
            # Call dropout for each layer
            indices = layer.dropout(indices)
            indices_active_nodes.append(indices)
            
        return indices_active_nodes
    
    
    def restore_matrices(self):
        """
        Restores the original weight matrices of layer after the dropout training
        
        """
        
        #Initialise output layer
        indices = range(self.layers[-1].w_memory.shape[1])
        
        for i in range(len(self.layers) - 1,0, - 1): # from last layer to the first layer

            # set layer 
            layer = self.layers[i]  
            # Reset each layer 
            indices = layer.reset_matrix(indices)

    def fit_mini_batch(self, X, y, toll, n_mini_batches = 32):
        """
        Trains the neural network with mini-batch gradient descent technique
        
        """
        
        self.L_history = [] # stores values of the loss 
        
        # Create batches of mini-batch training
        batch_size = int(len(X)/n_mini_batches) # batch dimension
        batches = [range(x,min(x+batch_size,len(X))) for x in range(0, len(X), batch_size)]
        
        for epoch in range(self.max_epochs):
            
            # Shuffle data
            shuffle_indices = np.arange(len(X))
            np.random.shuffle(shuffle_indices)
            X = X[shuffle_indices]
            y = y[shuffle_indices] 
            
            L_batch = []
            
            for batch in batches: # Train each batch independently 
                
                # Select batches
                X_batch = X[batch]
                y_batch = y[batch]
            
                # Train single batch
                L = self.train_single_batch(X_batch, y_batch)
                
                L_batch.append(L)
            
            self.L_history.append(np.mean(L_batch))
                
            # Stopping criterion
            if epoch > 3:
                if abs(self.L_history[-2] - self.L_history[-1]) <= toll and abs(self.L_history[-3] - self.L_history[-1]) <= toll and abs(self.L_history[-4] - self.L_history[-1]) <= toll:
                    print("Stop!")
                    break

        return self.L_history   
    
    def train_single_batch(self, X, y):
        """
        Trains the single mini batch from fit_mini_batch
        
        """
    
        active_indices = self.dropout_matrices()
        input_indices = active_indices[-1]
            
        io_layer = X[:,input_indices] # input/output of each layer BIAS
            
        # Compute the forward pass of the network --> y*
        for i in range(1,len(self.layers)):
                
            # compute output
            layer = self.layers[i]            
            io_layer = layer.forward_pass(io_layer)     
            
        # Compute the loss function and its derivative
        L = self.loss(io_layer, y)
        #self.L_history.append(L)
        dL = self.d_loss(io_layer, y) # this will be the first error to be transferred backwards
                
        io_error = dL # initialise the first gradient with the derivative of the loss

        for i in range(len(self.layers)-1,0,-1): # from last layer to the first layer
 
            # set layer 
            layer = self.layers[i]
            
            # update weights and compute error to pass at the previous layer
            io_error = layer.backward_pass(io_error,self.lr) #, opt = self.opt)
                
        self.restore_matrices()
        
        return L