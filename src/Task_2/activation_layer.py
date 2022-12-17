import numpy as np

class ActivationLayer():
    
    def __init__(self, type = ["sigmoid","ReLu","softmax"]):
        self.type = type
         
    def activation_function(self,x):
        """
        Computes the activation function
        
        """
        
        if self.type == "sigmoid":
            
            return np.exp(np.fmin(x, 0)) / (1 + np.exp(-np.abs(x)))  
       
        elif self.type == "ReLu":
                
            return np.maximum(0,x)
        
        elif self.type == "softmax":

            x = x.astype('float128')
            exp_vector = np.exp(x)
    
            return exp_vector / np.sum(exp_vector)
    
    def d_activation_function(self,x):
        """
        Computes the derivative of the activation function
        
        """        
        
        if self.type == "sigmoid":
            
            sig = np.exp(np.fmin(x, 0)) / (1 + np.exp(-np.abs(x))) 
            
            return sig*(1-sig)
        
        elif self.type == "ReLu":
                
            return (x > 0) * 1  
        
        elif self.type == "softmax":
    
            return np.ones_like(x)        
              