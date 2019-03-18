import torch
import torch.utils.data
import torch.nn as nn
import pandas as pd
import numpy as np
from time import ctime
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.interpolate import splev, splrep

p = 0.7
drop_out = torch.nn.Dropout(p)


 

CUDA = torch.cuda.is_available()
CUDA_DEVICE = 1
print("CUDA_DEVICE", CUDA_DEVICE)
if CUDA:
    torch.cuda.set_device(CUDA_DEVICE)

############ Loading Data######################	
input_data = pd.read_csv('/home/NewUsersDir/tmallava/pytorch/Data/gbm_data_exp_with_filtered_genes_greater_than_fifteen.csv', header = None)#GBM_data_bef_norm.csv, gbm_data_exp_with_filtered_genes

row_index = pd.read_csv("/home/NewUsersDir/tmallava/pytorch/Data/index_gbm_data_exp_with_filtered_genes.csv", header = None) #Indexes_for_data_BIBM.csv, index_gbm_data_exp_with_filtered_genes

months = pd.read_csv("/home/NewUsersDir/tmallava/pytorch/Data/original_months.csv", header = None)#original_months, original_months

adj_matrix_p = pd.read_csv("/home/NewUsersDir/tmallava/pytorch/Data/adjacency_matrix_with_998_pathways.csv", header = None) #adj_matrix_p,  adjacency_matrix_with_692_pathways

print("adj_matrix_p", adj_matrix_p.shape)
print("input_data",input_data.shape)
row_index_np = np.array(row_index)
months = np.array(months) 
input_data = np.array(input_data)

adj_matrix_p_trans = np.array(adj_matrix_p.T)
adj_matrix_tensor = torch.from_numpy(adj_matrix_p_trans)
adj_matrix_tensor = adj_matrix_tensor.type(torch.FloatTensor)
adj_matrix_tensor = adj_matrix_tensor.cuda()



def mask_vector1(weights, percentile):
     prob =  nn.Dropout((percentile/100))
     #print("prob", prob)
     weight_mask = prob(weights)
     #print(torch.nonzero(weight_mask==0))
     #print("weight_mask", weight_mask[0:5, 0:5])
     return weight_mask  


# def drop_out1(keep_prob, data):
#     D1 = torch.rand(data.shape[1])
#     bool_mat = D1 < keep_prob
#     bool_mat = bool_mat.type(torch.FloatTensor)
#     if CUDA:
#        bool_mat = bool_mat.cuda()
#     data_after_drop_out = (data * bool_mat )
#     data_after_drop_out = data_after_drop_out / keep_prob
#     #print("data_after_drop_out", data_after_drop_out[0:5, 0:5])
#     return data_after_drop_out

# def mask_vector(weights, percentile):
#     D1 = torch.rand(weights.shape[1])
#     bool_drop_connect = D1 < (1- percentile )
#     bool_drop_connect = bool_drop_connect.type(torch.FloatTensor)
#     bool_drop_connect = bool_drop_connect.cuda()
#     #print("D2", weight_after_drop_connect)
#     weight_mask = (weights * bool_drop_connect)  
#     #print("h_t", weight_mask)
#     return weight_mask 

def mask_vector(weights, percentile):
        
        weights_nodes = weights.cpu()
        weights_nodes = weights_nodes.data.numpy()
        absolute_weights = np.absolute(weights_nodes)
        
        weight_perc = np.percentile(absolute_weights, percentile)
        #print("weight_perc", weight_perc)
        
           
        absolute_weights[np.absolute(weights_nodes) < weight_perc] = 0
        absolute_weights[np.absolute(weights_nodes) >= weight_perc] = 1
        
        
        
        prob_weights = absolute_weights
        weight_mask = weights_nodes * prob_weights
        weight_mask = torch.from_numpy(weight_mask)
        weight_mask = weight_mask.cuda()
        #print("weight_mask", weight_mask[0:5, 0:5])
        #print("zeros in weight_mask", np.count_nonzero(weight_mask==0))
        return weight_mask


def interpolation(x ,y , perc_val):

     tck = splrep(x, y)
     all_err_val = splev(perc_val, tck).tolist()
     min_all_err_val_index = all_err_val.index(min(all_err_val))
     optimal_sparse = perc_val[min_all_err_val_index]
     min_error = all_err_val[min_all_err_val_index]

     return all_err_val, optimal_sparse,min_error


class RBM():

    def __init__(self, num_visible, num_hidden, k, learning_rate=[1e-3,0.5], 
                                    momentum_coefficient=0.5, weight_decay=1e-4, 
                                    use_cuda=True, percentile = [70,80,90]):

        super(RBM,self).__init__()
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.k = k
        self.learning_rate = learning_rate
        self.momentum_coefficient = momentum_coefficient
        self.weight_decay = weight_decay
        self.use_cuda = use_cuda
        self.percentile = percentile
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
        #print("self.weights", self.weights[0:5, 0:4])

    def sample_hidden(self, visible_probabilities):
	
        if self.weights.shape == adj_matrix_tensor.shape:
             
             self.weights = self.weights * adj_matrix_tensor
        else:
		
             self.weights = self.weights

        hidden_activations = torch.matmul(visible_probabilities, self.weights) + self.hidden_bias
        hidden_probabilities = self.sigmoid(hidden_activations)
        #print("hidden_probabilities", hidden_probabilities[0:5 , 0:5])
        return hidden_probabilities

    def sample_visible(self, hidden_probabilities):
        if self.weights.shape == adj_matrix_tensor.shape:
             
             #print("bef_self.weights", self.weights[0:5, 0:4])
             self.weights = self.weights * adj_matrix_tensor
             #print("self.weights", self.weights[0:5, 0:4])
        else:
		
             self.weights = self.weights
	
        visible_activations = torch.matmul(hidden_probabilities, self.weights.t()) + self.visible_bias

        visible_probabilities = self.sigmoid(visible_activations)# + a # self.sigmoid(visible_activations)
        return visible_probabilities
		
    def forward(self,input_data):
        'data->hidden'
        x = self.sample_hidden(input_data)
        #output = (x >= self.random_probabilities(self.num_hidden)).float()

        #np.savetxt('/home/NewUsersDir/tmallava/pytorch/Results/weights{0}.csv' .format((self.weights.shape)),self.weights.cpu().data.numpy(), delimiter = ',')
        
        return x
		
    def sample_gaussian(self, h_data):
    
        
        noise = torch.randn(h_data.shape)
        noise = noise.cuda()
        out_put = noise+h_data
        return out_put
 
    
    def contrastive_divergence(self, input_data):
	
        ############# Positive phase ############
        
        positive_hidden_probabilities = self.sample_hidden(input_data)
		
        positive_hidden_probabilities = drop_out(positive_hidden_probabilities)    #drop_out(positive_hidden_probabilities)

        positive_hidden_activations = (positive_hidden_probabilities >= self.random_probabilities(self.num_hidden)).float()
     
        positive_associations = torch.matmul(input_data.t(), positive_hidden_probabilities)#positive_hidden_activations

        ############# Negative phase ############
        hidden_activations = positive_hidden_probabilities #positive_hidden_activations


        for step in range(self.k):
            visible_probabilities = self.sample_visible(hidden_activations)
            #visible_probabilities = self.sample_gaussian(visible_probabilities)
            hidden_probabilities = self.sample_hidden(visible_probabilities)
            hidden_probabilities = drop_out(hidden_probabilities) #drop_out(hidden_probabilities)

            hidden_activations = (hidden_probabilities >= self.random_probabilities(self.num_hidden)).float()

        negative_visible_probabilities = visible_probabilities

        negative_hidden_probabilities = hidden_probabilities

        negative_hidden_activations = negative_hidden_probabilities # >= self.random_probabilities(self.num_hidden)).float()

        negative_associations = torch.matmul(negative_visible_probabilities.t(), negative_hidden_activations)#negative_hidden_probabilities

        ############# Update parameters ############

        self.weights_momentum *= self.momentum_coefficient
        self.weights_momentum += (positive_associations - negative_associations)
        
        self.visible_bias_momentum *= self.momentum_coefficient
        self.visible_bias_momentum += torch.sum(input_data - negative_visible_probabilities, dim=0)

        self.hidden_bias_momentum *= self.momentum_coefficient
        self.hidden_bias_momentum += torch.sum(positive_hidden_probabilities - negative_hidden_probabilities, dim=0)

        batch_size = input_data.size(0)
		

        self.weights +=  self.learning_rate * self.weights_momentum / batch_size
        #print("##################cd######################", self.learning_rate)
        self.visible_bias +=  self.learning_rate * self.visible_bias_momentum  / batch_size
        self.hidden_bias +=  self.learning_rate * self.hidden_bias_momentum / batch_size
        

        self.weights -= self.weights * self.weight_decay  ####### L2 weight decay ######
        #self.weights += self.learning_rate* (((self.weight_decay * self.weights)+self.weights_momentum)/ batch_size)
        #print("###########BEFORE_OPTIMAL_SPARSE_No.of Zeros", len((self.weights == 0).nonzero()))

        if self.weights.shape == adj_matrix_tensor.shape:
            #print("#############first layer##################")

            error_final = self.compute_reconstruction_error(input_data)
            #print("zeros in first layer", len((self.weights == 0).nonzero()))
            
        else:
            #cot = []
            #cot1 = []
            #orig_W = self.weights
            #print("before_optimal_self.weights", self.weights[0:5, 0:5])
            
            #for i in self.percentile:
                
                #self.weight = orig_W
                #new_weights = self.weight.clone()
                #self.weights = mask_vector1(new_weights,i)
                #print("self.weights", self.weights[0:5, 0:5])
               
                #error = self.compute_reconstruction_error(input_data)
                #cot.append(error) 

            #all_error_values, optimal_sparse_level, min_error = interpolation(self.percentile,cot,torch.linspace(min(self.percentile),max(self.percentile), 50)) 
			
            #print("min_error", min_error)
            #min_ynew = cot.index(min(cot))
            #optimal_sparse_level = self.percentile[min_ynew]
            #print("##############optimal_percentile############", optimal_sparse_level)
            #self.weights = mask_vector1(new_weights, optimal_sparse_level)
            
            error_final = self.compute_reconstruction_error(input_data)
            #print("error_final", error_final)
            #print("self.weights.shape", self.weights.shape)
            #print("No.of Zeros", len((self.weights == 0).nonzero()))           
            #np.savetxt('/home/NewUsersDir/tmallava/pytorch/Results/optimal_percentile{0}.csv' .format((len(optimal_percentile),time)),optimal_percentile, delimiter = ',')

        return error_final 
    
    
    def compute_reconstruction_error(self, data):
        """
        Computes the reconstruction error of the data.
        :param data: array-like, shape = (n_samples, n_features)
        :return:
        """
        
        data_transformed = self.sample_hidden(data) #self._compute_hidden_units_matrix(X)
        data_reconstructed = self.sample_visible(data_transformed) #self._compute_visible_units_matrix(transformed_data)
        mse = torch.sum((data_reconstructed - data) ** 2)
        l2_reg_term = 0.5 * (self.weight_decay *(torch.sum( self.weights ** 2)))
		
        #print("l2_reg_term", l2_reg_term)
        err = mse + l2_reg_term
        #print("error", err)
        return err

    def sigmoid(self, x):
        return 1 / (1 + torch.exp(-x))


    def random_probabilities(self, num):
        random_probabilities = torch.rand(num) 

        if self.use_cuda:
            random_probabilities = random_probabilities.cuda()

        return random_probabilities


    def train(self,train_data , num_epochs = 10,batch_size= 64):

        BATCH_SIZE = batch_size
        some_value = 5e+25
        high_err = []
        error_l = []
        if(isinstance(train_data ,torch.utils.data.DataLoader)):
            train_loader = train_data
            
        else:
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)      
       
        error = []
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
            #if epoch_err < some_value:
                #error_l.append(epoch_err)
            #else:
                #high_err.append(epoch_err	)
				
                #if len(high_err) >= 20:
                    #break
            #if len(high_err)>=2:
                #if high_err[-1]< high_err[-2]:
                   #some_value = high_err[-1]
                #else:
                   #some_value = error[-1]
            #else:
            #some_value = error[-1]
            
        #print("high_err", high_err)
        #print("len(error_l", len(error))
        #print("len(error_l", len(error_l))
        print(self.num_visible )
        plt.plot(error)
        print(ctime())
        time  = ctime()
        plt.savefig("/home/NewUsersDir/tmallava/pytorch/Results/error plot{0}.png" .format((self.num_visible ,time,[CUDA_DEVICE])))
        plt.show()
        plt.clf()
        return self.weights
     
    
class DBN():
    def __init__(self,
                num_visible = 256,
                num_hidden = [64 , 100],
                k = 2,
                learning_rate = [1e-5,0.5],
                momentum_coefficient = 0.5,
                weight_decay = 1e-4,
                use_cuda = False, percentile = [70,80,90]):  

        
        super(DBN,self).__init__()
        self.n_layers = len(num_hidden)
        print("no.of hidden layers", len(num_hidden))
        self.rbm_layers =[]
        self.percentile = percentile
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
                    use_cuda=use_cuda, percentile = self.percentile)

            self.rbm_layers.append(rbm)

        

		
    def train_static(self, train_data,train_labels,num_epochs,batch_size):
        '''
        Greedy Layer By Layer training
        Keeping previous layers as static
        '''
        train_data = train_data.cuda()
        tmp = train_data
        probabilities = []
        weights_all_layers = []
        print('Loading dataset...')
        for i in range(len(self.rbm_layers)):
            print("-" * 20)
            print("Training the rbm layer {}".format(i+1))
            print("************data_size***********:", tmp.shape)

            tensor_x = tmp.type(torch.FloatTensor) # transform to torch tensors
            tensor_y = train_labels.type(torch.FloatTensor)
            _dataset = torch.utils.data.TensorDataset(tensor_x,tensor_y) # create your datset
            _dataloader = torch.utils.data.DataLoader(_dataset, batch_size ) # create your dataloader
            
            weights = self.rbm_layers[i].train(_dataloader, num_epochs,batch_size)
            # print(train_data.shape)
            layer_data = tmp.view((tmp.shape[0] , -1)).type(torch.FloatTensor)#flatten
            layer_data = layer_data.cuda()
            layer_data = self.rbm_layers[i].forward(layer_data)
            layer_data = layer_data.cuda()
            probabilities.append(layer_data)
            #probabilities_tensor = torch.cat(probabilities)
            tmp = layer_data
            layer_data = layer_data.cpu()
            layer_data = layer_data.data.numpy()
            weights_all_layers.append(weights)
            print("layer_data.shape", layer_data[0:5, 0:5])

            time = ctime()
            #np.savetxt('/home/NewUsersDir/tmallava/pytorch/Results/probability{0}.csv' .format((layer_data.shape,time)),layer_data, delimiter = ',')
        return self, probabilities, layer_data, weights_all_layers


print("no. of samples",len(row_index_np)) 


for j in range (len(row_index_np)):
    
        index_data = [int (i) for i in row_index_np[j] ]
        print("######################column index#############################",j)



        train_array_bef_norm_withlabels = input_data[index_data] 
        train_array_bef_norm = train_array_bef_norm_withlabels
        normalized_X = (train_array_bef_norm- np.mean(train_array_bef_norm, axis = 0))/np.std(train_array_bef_norm, axis = 0)
        train_array = normalized_X        
        print(train_array.shape)
               
        months_index = months[index_data]
        #print("months_index", months_index)
        features = torch.from_numpy(train_array)
        labels = torch.from_numpy(months_index) 
        #print(train_array.shape)                  

        print('Training RBM...')
        dbn = DBN(num_visible = input_data.shape[1], num_hidden = [adj_matrix_p.shape[0], 500, 200, 2], k = 1, 
                          learning_rate = [0.0005, 0.05, 0.05, 0.005], momentum_coefficient = 0.2, 
                          weight_decay = 1e-4, use_cuda = True, percentile = [10,20,30,40,50,60,70,80,90,99]) #0.0005, 0.01, 0.01, 0.007, momentum_coefficient = 0.1

        ab, all_prob, model, weights = dbn.train_static(features,labels,  1500, 24) # 5000 , 24
        #np.savetxt('/home/NewUsersDir/tmallava/pytorch/Results/weights{0}.csv' .format((model.shape,j)),model, delimiter = ',')
        #np.savetxt('/home/NewUsersDir/tmallava/pytorch/Results/weights_CUDA_DEVICE{0}.csv' .format((weights[-1].shape[1],j,[CUDA_DEVICE])),weights[-1].cpu().numpy(), delimiter = ',')
        #np.savetxt('/home/NewUsersDir/tmallava/pytorch/Results/weights_CUDA_DEVICE{0}.csv' .format((weights[-2].shape[1],j, [CUDA_DEVICE])),weights[-2].cpu().numpy(), delimiter = ',')
        #np.savetxt('/home/NewUsersDir/tmallava/pytorch/Results/weights_CUDA_DEVICE{0}.csv' .format((weights[-3].shape[1],j, [CUDA_DEVICE])),weights[-3].cpu().numpy(), delimiter = ',')
        #np.savetxt('/home/NewUsersDir/tmallava/pytorch/Results/weights_CUDA_DEVICE{0}.csv' .format((weights[-4].shape[1],j, [CUDA_DEVICE])),weights[-4].cpu().numpy(), delimiter = ',')
        np.savetxt('/home/NewUsersDir/tmallava/pytorch/Results/CL_CUDA_DEVICE{0}.csv' .format((all_prob[-1].shape[1],j,[CUDA_DEVICE])),all_prob[-1].cpu().numpy(), delimiter = ',')
        np.savetxt('/home/NewUsersDir/tmallava/pytorch/Results/HL2_CUDA_DEVICE{0}.csv' .format((all_prob[-2].shape[1],j, [CUDA_DEVICE])),all_prob[-2].cpu().numpy(), delimiter = ',')
        np.savetxt('/home/NewUsersDir/tmallava/pytorch/Results/HL1_CUDA_DEVICE{0}.csv' .format((all_prob[-3].shape[1],j, [CUDA_DEVICE])),all_prob[-3].cpu().numpy(), delimiter = ',')
        np.savetxt('/home/NewUsersDir/tmallava/pytorch/Results/PL_CUDA_DEVICE{0}.csv' .format((all_prob[-4].shape[1],j, [CUDA_DEVICE])),all_prob[-4].cpu().numpy(), delimiter = ',')
        for i in range (2, 5):

           print("##############################cluster_number############################", i)
           dbn = DBN(num_visible = all_prob[-2].shape[1], num_hidden = [i], k = 1, 
           learning_rate = [0.005], momentum_coefficient = 0.2, 
           weight_decay = 1e-4, use_cuda = True, percentile = [10,20,30,40,50,60,70,80,90,99]) 
           ab, cl_prob, model, weights = dbn.train_static(all_prob[-2],labels, 1500, 24)
           #cl_prob = cl_prob[-1].cpu() 
           #model = model.data.numpy() 
           time = ctime()
           np.savetxt('/home/NewUsersDir/tmallava/pytorch/Results/probability{0}.csv' .format((cl_prob[-1].shape, j,[CUDA_DEVICE])),cl_prob[-1].cpu().numpy(), delimiter = ',')
           #np.savetxt('/home/NewUsersDir/tmallava/pytorch/Results/weights_CUDA_DEVICE{0}.csv' .format((weights[-1].shape, j,[CUDA_DEVICE])),weights[-1].cpu().numpy(), delimiter = ',')
           print("num_hidden", cl_prob[-1][0:5, 0:5])       
#np.savetxt("/home/NewUsersDir/tmallava/pytorch/Results/model_parametrs.txt", dbn)
import inspect

def DBN_args(num_visible = input_data.shape[1], num_hidden = [adj_matrix_p.shape[0], 1500, 200, 2], k = 3, learning_rate = [5e-4, 5e-2, 5e-2, 5e-4], momentum_coefficient = 0.1, weight_decay = 1e-5, use_cuda = True, epochs = 5000, batchsize = 24):
    pass

args = inspect.getargspec(DBN_args)

np.savetxt("/home/NewUsersDir/tmallava/pytorch/Results/model_parametrs{0}.txt".format([CUDA_DEVICE]), args, fmt = '%18s')