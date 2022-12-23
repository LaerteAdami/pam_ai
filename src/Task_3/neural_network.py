from sklearn.metrics import accuracy_score
import torch
import matplotlib.pyplot as plt

class NeuralNetwork():
    
    def __init__(self, nn):
        self.nn = nn
    
    def initialise(self, loss_function, optimizer):
        """
        Initialise the neural network with the loss function and the optimizer to be used
      
    
        """
    
        self.loss_function = loss_function
        self.optimizer = optimizer
        
    def fit(self, data_loader, validation_loader, max_epochs):
        """
        Fit the neural network and validate the model thourgh validation loss and validation accuracy
      
        """
        
        L_history = []
        L_history_val = []
        acc_history = []
                       
        for epoch in range(max_epochs):
            
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')

            # model in training mode
            self.nn.train()
            
            # Initialise batch of each loss
            L_batch = 0.0

            for data, target in data_loader: # Loop for each batch
                
                data = data.to(device)
                target = target.to(device)
            
                # Set gradients to zero
                self.optimizer.zero_grad()
    
                # Forward pass
                output = self.nn(data)
    
                # Compute loss 
                L = self.loss_function(output, target)
    
                # Backward pass
                L.backward()
        
                # Weights update
                self.optimizer.step()
                
                L_batch += L.item() * data.size(0)
                
            L_history.append(L_batch/len(data_loader.sampler))


            ### VALIDATION ###
            self.nn.eval()

            # Initialise batch of each loss
            L_batch_val = 0.0

            # Validate the model
            for data, target in validation_loader:

                data = data.to(device)
                target = target.to(device)

                if torch.cuda.is_available():
                    data = data.cuda()

                output = self.nn(data)

                L_val = self.loss_function(output, target)

                L_batch_val += L_val.item() * data.size(0)

            # Store variable for validation loss and accuracy
            L_history_val.append(L_batch_val/len(validation_loader.sampler))
  
        # Store the losses and accuracy as a class attribute
        self.L_history = L_history
        self.L_history_val = L_history_val
        
    def evaluate(self, data_loader, plot_flag):
        """
        Evaluate the model by computing the accuracy of predictions and plotting the loss vs epochs
      
        """
        
        # Model in eval mode
        self.nn.eval()
        
        # Initialise target tensors (test and predict)
        y_pred = torch.LongTensor()
        y_test = torch.LongTensor()
        
        for data, target in data_loader:

            if torch.cuda.is_available():
                data = data.cuda()
            
            batch_output = self.nn(data)
            
            # Take the index with highest probability
            batch_preds = batch_output.cpu().data.max(1,keepdim=True)[1]
            
            #Combine tensors from each batch
            y_pred = torch.cat((y_pred, batch_preds), dim=0) 
            y_test = torch.cat((y_test, target), dim=0)

      
        # Print accuracy and plot loss when requested
        if plot_flag:
            
            print("Accuracy: %.2f"%accuracy_score(y_pred, y_test))

            fig, ax = plt.subplots()
            ax.plot(self.L_history,label="train")
            ax.plot(self.L_history_val,label="val")
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Loss")
            ax.legend();
            
        return y_pred, y_test  
    
    def reset_model(self):
        """
        Reset the model. Source: https://discuss.pytorch.org/t/how-to-re-set-alll-parameters-in-a-network/20819
    
        """
        for layers in self.nn.children():
            for layer in layers:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        
