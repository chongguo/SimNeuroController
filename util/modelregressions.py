import sklearn 
import torch
from tqdm import tqdm
import numpy as np
from sklearn.linear_model import LinearRegression
from controllers.neuralcontroller import NeuralController

class CNNCrossFit():
    def __init__(self,target_net,source_net,cnn_layer,target_units,source_units):
        self.target_unit = target_units
        self.source_unit = source_units
        self.n_tar = len(target_units)
        self.n_sor = len(source_units)
        self.cnn_layer = cnn_layer
        self.batch_size = 100;
        # this is only for the size of image used for control, model could be fit with any input size
        self.img_dim = [256,256,3]
        # both nets are truncated to the selected layer and set to evaluation mode
        self.target_net = target_net[:cnn_layer]
        self.source_net = source_net[:cnn_layer]
        self.target_net.eval()
        self.source_net.eval()
    
    def design(self,dataset,ntrain,ntest):
        # create train and test split for evaluating model fit
        self.ndata = len(dataset)
        self.ntrain = ntrain
        self.ntest = ntest
        [train_set, test_set] = torch.utils.data.random_split(dataset,[self.ntrain,self.ntest])
        # this parameter controls the spatial subsampling for fitting
        self.subsamp = 10
        # create a train and a test data loader
        self.trainloader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size,
                        shuffle=False, num_workers=0)
        self.testloader = torch.utils.data.DataLoader(test_set, batch_size=self.batch_size,
                        shuffle=False, num_workers=0)
        # create empty list for holding data
        Xsor_train = []
        Xtar_train = []
        Xsor_test = []
        Xtar_test = []
        # assemble data matix for training and test set
        for [i, trainbatch] in enumerate(tqdm(self.trainloader)):
            xsor = self.source_net.forward(trainbatch)
            xsor = xsor[:,self.source_unit,::10,::10].permute(1,0,2,3).reshape((self.n_sor,-1)).detach().numpy()
            xtar = self.target_net.forward(trainbatch)
            xtar = xtar[:,self.target_unit,::10,::10].permute(1,0,2,3).reshape((self.n_tar,-1)).detach().numpy()
            Xsor_train.append(xsor) 
            Xtar_train.append(xtar) 
        for [i, testbatch] in enumerate(tqdm(self.testloader)):
            xsor = self.source_net.forward(testbatch)
            xsor = xsor[:,self.source_unit,::10,::10].permute(1,0,2,3).reshape((self.n_sor,-1)).detach().numpy()
            xtar = self.target_net.forward(testbatch)
            xtar = xtar[:,self.target_unit,::10,::10].permute(1,0,2,3).reshape((self.n_tar,-1)).detach().numpy()
            Xsor_test.append(xsor) 
            Xtar_test.append(xtar) 
        
        self.Xsor_train = Xsor_train
        self.Xsor_test = Xsor_test
        self.Xtar_train = Xtar_train
        self.Xtar_test = Xtar_test
        
    def fit(self,option = 'LR'):
        # options:
        # LR - linear regression for each target unit independently
        # CCA - communicate through a smaller shared subspace
        X_train = np.concatenate(self.Xsor_train,axis=1).T;
        X_test = np.concatenate(self.Xsor_test,axis=1).T;
        y_train = np.concatenate(self.Xtar_train,axis=1).T;
        y_test = np.concatenate(self.Xtar_test,axis=1).T;
        self.w = np.empty((self.n_sor,self.n_tar))
        self.b = np.empty(self.n_tar)
        self.train_score = [None]*self.n_tar
        self.test_score = [None]*self.n_tar
        
        for i in range(self.n_tar):
            if option == 'LR':
                reg = LinearRegression().fit(X_train, y_train[:,i])
            self.w[:,i] = reg.coef_
            self.b[i] =reg.intercept_
            self.train_score[i] = reg.score(X_train, y_train[:,i])
            self.test_score[i] = reg.score(X_test, y_test[:,i])
    
    def control(self):
        # implement control with either source or target net
        self.target_act = np.zeros(self.n_tar)
        self.source_act = np.zeros(self.n_tar)
        self.ctr_score = np.zeros(self.n_tar)
        for i in tqdm(range(self.n_tar)):
            # target control
            target_controller = NeuralController(self.target_net,self.target_unit[i],self.img_dim,)
            target_controller.visualize(niter=30,label = 'snet_tar_l'+str(self.cnn_layer)+'_u'+str(self.target_unit[i]))
            self.target_act[i] = target_controller.act
            # source control
            source_controller = NeuralController(self.source_net,self.source_unit,self.img_dim,self.w[:,i],self.b[i])
            source_controller.visualize(niter=30,label = 'snet_source_l'+str(self.cnn_layer)+'_u'+str(self.target_unit[i]))
            # calculate fraction control
            x = self.target_net.forward(source_controller.processed_image)
            self.source_act[i] = torch.mean(x[0,self.target_unit[i]])
            self.ctr_score[i] = self.source_act[i]/(self.target_act[i]+.01)
        
                