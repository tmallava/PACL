import torch
import torch.utils.data
import torch.nn as nn
import pandas as pd
import numpy as np
from time import ctime
import matplotlib.pyplot as plt


p = 0.7
drop_out = torch.nn.Dropout(p)
 
CUDA = torch.cuda.is_available()
CUDA_DEVICE = 1
print("CUDA_DEVICE", CUDA_DEVICE)
if CUDA:
    torch.cuda.set_device(CUDA_DEVICE)


##### initializing the RBM model ############

class RBM():

    def __init__(self, num_visible, num_hidden, k, learning_rate=[1e-3,0.5], 
                                    momentum_coefficient=0.5, weight_decay=1e-4, 
                                    use_cuda=True):

        super(RBM,self).__init__()
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.k = k
        self.learning_rate = learning_rate
        self.momentum_coefficient = momentum_coefficient
        self.weight_decay = weight_decay
        self.use_cuda = use_cuda
        self.weights = torch.randn(num_visible, num_hidden) 
        self.visible_bias = torch.ones(num_visible) 
        self.hidden_bias = torch.zeros(num_hidden)

        self.weights_momentum = torch.zeros(num_visible, num_hidden)
        self.visible_bias_momentum = torch.zeros(num_visible)
        self.hidden_bias_momentum = torch.zeros(num_hidden)
       
        if self.use_cuda:
            self.weights = self.weights.cuda()
            self.visible_bias = self.visible_bias.cuda()
            self.hidden_bias = self.hidden_bias.cuda()

            self.weights_momentum = self.weights_momentum.cuda() 
            self.visible_bias_momentum = self.visible_bias_momentum.cuda()
            self.hidden_bias_momentum = self.hidden_bias_momentum.cuda()
       
###### sampling hidden units from visible data  #########
    def sample_hidden(self, visible_probabilities):
	if self.weights.shape == adj_matrix_tensor.shape:
           self.weights = self.weights * adj_matrix_tensor
        else:
	   self.weights = self.weights
        hidden_activations = torch.matmul(visible_probabilities, self.weights) + self.hidden_bias
        hidden_probabilities = self.sigmoid(hidden_activations)
        return hidden_probabilities

###### sampling visible units from hidden data  #########
    def sample_visible(self, hidden_probabilities):
        if self.weights.shape == adj_matrix_tensor.shape:
           self.weights = self.weights * adj_matrix_tensor
        else:
	   self.weights = self.weights
	visible_activations = torch.matmul(hidden_probabilities, self.weights.t()) + self.visible_bias
        visible_probabilities = self.sigmoid(visible_activations)
        return visible_probabilities
	
	
    def forward(self,input_data):
        'data->hidden'
        x = self.sample_hidden(input_data)
        return x
		
    
    def contrastive_divergence(self, input_data):
	
        ############# Positive phase ############
        
        positive_hidden_probabilities = self.sample_hidden(input_data)
	positive_hidden_probabilities = drop_out(positive_hidden_probabilities)    
        positive_hidden_activations = positive_hidden_probabilities
        positive_associations = torch.matmul(input_data.t(), positive_hidden_probabilities)

        ############# Negative phase ############
	
        hidden_activations = positive_hidden_probabilities 

        for step in range(self.k): #### CD step
            visible_probabilities = self.sample_visible(hidden_activations)
            hidden_probabilities = self.sample_hidden(visible_probabilities)
            hidden_probabilities = drop_out(hidden_probabilities) 
            hidden_activations = hidden_probabilities 

        negative_visible_probabilities = visible_probabilities
        negative_hidden_probabilities = hidden_probabilities
        negative_hidden_activations = negative_hidden_probabilities 
        negative_associations = torch.matmul(negative_visible_probabilities.t(), negative_hidden_activations)

        ############# Update parameters ############

        self.weights_momentum *= self.momentum_coefficient
        self.weights_momentum += (positive_associations - negative_associations)
        
        self.visible_bias_momentum *= self.momentum_coefficient
        self.visible_bias_momentum += torch.sum(input_data - negative_visible_probabilities, dim=0)

        self.hidden_bias_momentum *= self.momentum_coefficient
        self.hidden_bias_momentum += torch.sum(positive_hidden_probabilities - negative_hidden_probabilities, dim=0)

        batch_size = input_data.size(0)
		

        self.weights +=  self.learning_rate * self.weights_momentum / batch_size
        self.visible_bias +=  self.learning_rate * self.visible_bias_momentum  / batch_size
        self.hidden_bias +=  self.learning_rate * self.hidden_bias_momentum / batch_size
        self.weights -= self.weights * self.weight_decay  ####### L2 weight decay ######
        error_final = self.compute_reconstruction_error(input_data)
        return error_final 
    
 ############ compute reconstruction of the visible data and the error  
    
    def compute_reconstruction_error(self, data):
        data_transformed = self.sample_hidden(data) 
        data_reconstructed = self.sample_visible(data_transformed) 
        mse = torch.sum((data_reconstructed - data) ** 2)
        l2_reg_term = 0.5 * (self.weight_decay *(torch.sum( self.weights ** 2)))
        err = mse + l2_reg_term
        return err

    def sigmoid(self, x):
        return 1 / (1 + torch.exp(-x))


    def random_probabilities(self, num):
        random_probabilities = torch.rand(num) 
        if self.use_cuda:
            random_probabilities = random_probabilities.cuda()
        return random_probabilities

####### training of the RNM  model ###########

    def train(self,train_data , num_epochs = 10,batch_size= 64):

        BATCH_SIZE = batch_size
        some_value = 5e+25
        high_err = []
        error = []
        if(isinstance(train_data ,torch.utils.data.DataLoader)):
            train_loader = train_data
        else:
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)      
       
        
        self.lr_decay = self.learning_rate / num_epochs
        for epochs in range(num_epochs):
            epoch_err = 0.0;
            for batch,label in train_loader: #
                if CUDA:
                    batch = batch.cuda()                   
                batch_err = self.contrastive_divergence(batch)
                epoch_err += batch_err
            self.learning_rate = self.learning_rate * 1/(1 + self.lr_decay * epochs)
            if epochs % 100 == 0:
               print("Epoch Error(epoch:%d) : %.4f" % (epochs , epoch_err),ctime())

            error.append(epoch_err) 
        plt.plot(error)
        plt.savefig("/error plot{0}.png" .format((self.num_visible ,time,[CUDA_DEVICE])))
        plt.show()
        plt.clf()
        return self.weights
     
 
########### Initializung DBN Framework #############
class DBN():
    def __init__(self,
                num_visible = 256,
                num_hidden = [64 , 100],
                k = 2,
                learning_rate = [1e-5,0.5],
                momentum_coefficient = 0.5,
                weight_decay = 1e-4,
                use_cuda = False):  

        
        super(DBN,self).__init__()
        self.n_layers = len(num_hidden)
        print("no.of hidden layers", len(num_hidden))
        self.rbm_layers =[]
        self.learning_rate = learning_rate

############### Creating different RBM layers ###############

        for i in range(self.n_layers):
            input_size = 0
            
            if i==0:
                input_size = num_visible
            else:
                input_size = num_hidden[i-1]
            print("input_size:",input_size)
            print("hidden_size:",num_hidden[i])
            rbm = RBM(num_visible = input_size,
                    num_hidden = num_hidden[i],
                    k= k,
                    learning_rate = learning_rate[i],
                    momentum_coefficient = momentum_coefficient,
                    weight_decay = weight_decay,
                    use_cuda=use_cuda)

            self.rbm_layers.append(rbm)

     		
    def train_static(self, train_data,train_labels,num_epochs,batch_size):
        '''
        Greedy Layer By Layer training by keeping previous layers as static
        '''
        train_data = train_data.cuda()
        tmp = train_data
        all_layers_probabilities = []
        weights_all_layers = []
        
        for i in range(len(self.rbm_layers)):
            print("Training the rbm layer {}".format(i+1))
            print("************data_size***********:", tmp.shape)

            tensor_x = tmp.type(torch.FloatTensor) # transform to torch tensors
            tensor_y = train_labels.type(torch.FloatTensor)
            _dataset = torch.utils.data.TensorDataset(tensor_x,tensor_y) # create your datset
            _dataloader = torch.utils.data.DataLoader(_dataset, batch_size ) # create your dataloader
            
            weights = self.rbm_layers[i].train(_dataloader, num_epochs,batch_size)
            
            layer_data = tmp.view((tmp.shape[0] , -1)).type(torch.FloatTensor)#flatten
            layer_data = layer_data.cuda()
            layer_data = self.rbm_layers[i].forward(layer_data)
            layer_data = layer_data.cuda()
            all_layers_probabilities.append(layer_data)
            
            tmp = layer_data
            layer_data = layer_data.cpu()
            layer_data = layer_data.data.numpy()
            weights_all_layers.append(weights)
            print("layer_data.shape", layer_data[0:5, 0:5])
        return all_layers_probabilities


############ Loading Data######################	
'path: path to input files'
'all the input files must be in csv file format'

input_data = pd.read_csv('path+ /gene_expression_data.csv', header = None) # input data has gene expression data and survival months
adj_matrix = pd.read_csv('path + /adjacency_matrix.csv', header = None) # Load a bi-adjacency matrix of pathways and genes

normalized_X = (input_data - np.mean(input_data, axis = 0))/np.std(input_data, axis = 0)
train_array = normalized_X        
#print(train_array.shape)
features = torch.from_numpy(train_array)
labels = torch.from_numpy(months_index) 

print('Training RBM...')

dbn = DBN(num_visible = input_data.shape[1], num_hidden = [adj_matrix_p.shape[0], 500, 200, 2], k = 2, 
                          learning_rate = [0.0005, 0.05, 0.05, 0.005], momentum_coefficient = 0.2, 
                          weight_decay = 1e-4, use_cuda = True) 

all_prob = dbn.train_static(gene_exp_data,  1500,  24) 
        
