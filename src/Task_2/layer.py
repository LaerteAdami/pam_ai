import numpy as np
import random

class Layer():
    
    def __init__(self,n_nodes, dropout_percentile = 1, activation_function = None):
        
        self.n = n_nodes 
        self.drop_perc = dropout_percentile
        self.af = activation_function
        self.w = None # to be initalised in the NeuralNetwork class
        self.b = None # to be initalised in the NeuralNetwork class
        self.optimizer = None # to be initalised in the NeuralNetwork class
           
        # For Adam Optimizer
        self.w_moment1 = None # to be initalised in the NeuralNetwork class
        self.w_moment2 = None # to be initalised in the NeuralNetwork class
        self.b_moment1 = None # to be initalised in the NeuralNetwork class
        self.b_moment2 = None # to be initalised in the NeuralNetwork class
    
    def forward_pass(self, X):
        
        self.input = X 
        
        z = np.dot(X,self.w) + self.b # type: ignore # z = w * X
        
        self.output = z
        
        if self.af is None:
            return z
        else:
            return self.af.activation_function(z) # a = f(z) where f is the activation function
    
    
    def backward_pass(self,output_error, learning_rate, opt = None):
        
        dL_da = output_error # gradient error from the previous layer
        
        # Derivative of the activation function
        if self.af is None:
            da_dz = np.ones_like(dL_da)
        else:
            da_dz = self.af.d_activation_function(self.output)
        
        input_error = dL_da * da_dz   
        
        # Source: Neural Network from scratch in Python
        dbias = np.sum(output_error, axis=0, keepdims= True)
        dweights = np.dot(self.input.T,input_error)
                
        # update weights of the layer
        if self.optimizer:
            # With optimizer
            self.w_moment1, self.w_moment2, dw = self.optimizer.update_weigths(dweights, self.w_moment1, self.w_moment2)# type: ignore
            self.b_moment1, self.b_moment2, db = self.optimizer.update_bias(dbias, self.b_moment1, self.b_moment2)# type: ignore
        else:
            dw = dweights * learning_rate
            db = dbias * learning_rate
    
        self.w -= dw 
        self.b -= db 
        
        input_error = np.dot(output_error, self.w.T) # gradient error for the next layer
        
        return input_error
     
    def dropout(self, next_indices = None):
        
        self.b_memory = self.b.copy() # type: ignore
        self.w_memory = self.w.copy() # type: ignore
        self.n_memory = self.n

        # ADAM
        self.b_moment1_memory = self.b_moment1.copy() # type: ignore
        self.b_moment2_memory = self.b_moment2.copy() # type: ignore       
        self.w_moment1_memory = self.w_moment1.copy() # type: ignore 
        self.w_moment2_memory = self.w_moment2.copy() # type: ignore 
            
        N = self.w.shape[0] # type: ignore
        
        active_nodes = int(N * self.drop_perc)
        indices = sorted(random.sample(range(0, N), active_nodes))
        
        self.w = self.w[indices][:,next_indices]# type: ignore # drop nodes from this and the next layer        
        self.n = indices
        self.b = self.b[:,next_indices] # type: ignore
        
        # ADAM
        self.w_moment1 = self.w_moment1[indices][:,next_indices]# type: ignore
        self.w_moment2 = self.w_moment2[indices][:,next_indices]# type: ignore
        self.b_moment1 = self.b_moment1[:,next_indices]# type: ignore
        self.b_moment2 = self.b_moment2[:,next_indices]# type: ignore
        
        return indices
  
    def reset_matrix(self, previous_indices):
        
        index_rows = self.n # active in this layer 
        index_columns = previous_indices # active in the previous layer 
        
        temp = self.w_memory[index_rows]
        temp[:,index_columns] = self.w
        self.w_memory[index_rows] = temp
        
        self.n = self.n_memory
        self.w = self.w_memory.copy()
        self.b = self.b_memory.copy()
        
        # ADAM
        self.b_moment1 = self.b_moment1_memory.copy()
        self.b_moment2 = self.b_moment2_memory.copy()
        self.w_moment1 = self.w_moment1_memory.copy()
        self.w_moment2 = self.w_moment2_memory.copy()        
        
        return index_rows    