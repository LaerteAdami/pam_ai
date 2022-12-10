import numpy as np

class ActivationLayer():
    
    def __init__(self, type = ["sigmoid","ReLu","softmax"]):
        self.type = type
         
    def activation_function(self,x):
        """
        Computes the activation function
        
        """
        
        if self.type == "sigmoid":
            
            #return 1 / (1 + np.exp(-x))
            return np.exp(np.fmin(x, 0)) / (1 + np.exp(-np.abs(x)))  # type: ignore 
       
        elif self.type == "ReLu":
                
            return np.maximum(0,x)  # type: ignore 
        
        elif self.type == "softmax":
            x = x.astype('float128')
            exp_vector = np.exp(x)  # type: ignore 
    
            return exp_vector / np.sum(exp_vector)  # type: ignore 
    
    def d_activation_function(self,x):
        """
        Computes the derivative of the activation function
        
        """        
        
        if self.type == "sigmoid":
            
            #sig = 1 / (1 + np.exp(-x))
            sig = np.exp(np.fmin(x, 0)) / (1 + np.exp(-np.abs(x)))  # type: ignore 
            
            return sig*(1-sig)
        
        elif self.type == "ReLu":
                
            return (x > 0) * 1  
        
        elif self.type == "softmax":
    
            return np.ones_like(x)        
              