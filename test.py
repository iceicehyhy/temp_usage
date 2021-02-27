import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from random import randint
import utils
import time
from utils import check_cifar_dataset_exists




class VGG_convnet(nn.Module):

    def __init__(self):

        super(VGG_convnet, self).__init__()

        # block 1:         3 x 32 x 32 --> 64 x 16 x 16        
        self.conv1a = nn.Conv2d(3,   64,  kernel_size=3, padding=1 )
        self.conv1b = nn.Conv2d(64,  64,  kernel_size=3, padding=1 )
        self.pool1  = nn.MaxPool2d(2,2)

        # block 2:         64 x 16 x 16 --> 128 x 8 x 8
        self.conv2a = nn.Conv2d(64,  128, kernel_size=3, padding=1 )
        self.conv2b = nn.Conv2d(128, 128, kernel_size=3, padding=1 )
        self.pool2  = nn.MaxPool2d(2,2)

        # block 3:         128 x 8 x 8 --> 256 x 4 x 4        
        self.conv3a = nn.Conv2d(128, 256, kernel_size=3, padding=1 )
        self.conv3b = nn.Conv2d(256, 256, kernel_size=3, padding=1 )
        self.pool3  = nn.MaxPool2d(2,2)
        
        #block 4:          256 x 4 x 4 --> 512 x 2 x 2
        self.conv4a = nn.Conv2d(256, 512, kernel_size=3, padding=1 )
        self.pool4  = nn.MaxPool2d(2,2)

        # linear layers:   512 x 2 x 2 --> 2048 --> 4096 --> 4096 --> 10
        self.linear1 = nn.Linear(2048, 4096)
        self.linear2 = nn.Linear(4096,4096)
        self.linear3 = nn.Linear(4096, 10)


    def forward(self, x):

        # block 1:         3 x 32 x 32 --> 64 x 16 x 16
        x = self.conv1a(x)
        x = F.relu(x)
        x = self.conv1b(x)
        x = F.relu(x)
        x = self.pool1(x)

        # block 2:         64 x 16 x 16 --> 128 x 8 x 8
        x = self.conv2a(x)
        x = F.relu(x)
        x = self.conv2b(x)
        x = F.relu(x)
        x = self.pool2(x)

        # block 3:         128 x 8 x 8 --> 256 x 4 x 4
        x = self.conv3a(x)
        x = F.relu(x)
        x = self.conv3b(x)
        x = F.relu(x)
        x = self.pool3(x)

        #block 4:          256 x 4 x 4 --> 512 x 2 x 2
        x = self.conv4a(x)
        x = F.relu(x)
        x = self.pool4(x)

        # linear layers:   512 x 2 x 2 --> 2048 --> 4096 --> 4096 --> 10
        x = x.view(-1, 2048)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x) 
        
        return x
    
    def return_name(self):
        return 'VGG_convnet'

def eval_on_test_set():

    running_error=0
    num_batches=0

    for i in range(0,10000,bs):

        minibatch_data =  test_data[i:i+bs]
        minibatch_label= test_label[i:i+bs]

        minibatch_data=minibatch_data.to(device)
        minibatch_label=minibatch_label.to(device)
        
        inputs = (minibatch_data - mean)/std

        scores=net( inputs ) 

        error = utils.get_error( scores , minibatch_label)

        running_error += error.item()

        num_batches+=1

    total_error = running_error/num_batches
    print( 'error rate on test set =', total_error*100 ,'percent')

net=VGG_convnet()

print(net)
#utils.display_num_param(net)
net.named_parameters()
bs = 128

def train(model):

    data_path=check_cifar_dataset_exists()
    device= torch.device("cuda")
    print(device)

    train_data=torch.load(data_path+'cifar/train_data.pt')
    train_label=torch.load(data_path+'cifar/train_label.pt')
    test_data=torch.load(data_path+'cifar/test_data.pt')
    test_label=torch.load(data_path+'cifar/test_label.pt')

    print(train_data.size())
    print(test_data.size())

    mean= train_data.mean()
    std= train_data.std()

    net = net.to(device)

    mean = mean.to(device)

    std = std.to(device)

    criterion = nn.CrossEntropyLoss()
    my_lr=0.25 
    bs= 128

    start=time.time()

    for epoch in range(1,20):
        
        # divide the learning rate by 2 at epoch 10, 14 and 18
        if epoch==10 or epoch == 14 or epoch==18:
            my_lr = my_lr / 2
        
        # create a new optimizer at the beginning of each epoch: give the current learning rate.   
        optimizer=torch.optim.SGD( net.parameters() , lr=my_lr )
            
        # set the running quatities to zero at the beginning of the epoch
        running_loss=0
        running_error=0
        num_batches=0
        
        # set the order in which to visit the image from the training set
        shuffled_indices=torch.randperm(50000)
    
        start=time.time()
        
        for count in range(0,50000,bs):
        
            # Set the gradients to zeros
            optimizer.zero_grad()
            
            # create a minibatch       
            indices=shuffled_indices[count:count+bs]
            minibatch_data =  train_data[indices]
            minibatch_label=  train_label[indices]
            
            # send them to the gpu
            minibatch_data=minibatch_data.to(device)
            minibatch_label=minibatch_label.to(device)
            
            # normalize the minibatch (this is the only difference compared to before!)
            inputs = (minibatch_data - mean)/std
            
            # tell Pytorch to start tracking all operations that will be done on "inputs"
            inputs.requires_grad_()

            # forward the minibatch through the net 
            scores=net( inputs ) 

            # Compute the average of the losses of the data points in the minibatch
            loss =  criterion( scores , minibatch_label) 
            
            # backward pass to compute dL/dU, dL/dV and dL/dW   
            loss.backward()

            # do one step of stochastic gradient descent: U=U-lr(dL/dU), V=V-lr(dL/dU), ...
            optimizer.step()
            

            # START COMPUTING STATS
            
            # add the loss of this batch to the running loss
            running_loss += loss.detach().item()
            
            # compute the error made on this batch and add it to the running error       
            error = utils.get_error( scores.detach() , minibatch_label)
            running_error += error.item()
            
            num_batches+=1        
        
        
        # compute stats for the full training set
        total_loss = running_loss/num_batches
        total_error = running_error/num_batches
        elapsed = (time.time()-start)/60
        

        print('epoch=',epoch, '\t time=', elapsed,'min','\t lr=', my_lr  ,'\t loss=', total_loss , '\t error=', total_error*100 ,'percent')
        eval_on_test_set() 
        print(' ')
        
        if (epoch % 9 == 0):
            if isinstance(net, torch.nn.DataParallel):
                net = net.module
            torch.save({
                'epoch': epoch,
                'optim_state_dict': optimizer.state_dict(),
                'model_state_dict': net.state_dict()}, net.return_name() + 'ckpt.pth'
            )


def load_pretrained_weight(model, weight_path):
    print ("Loading pre-trained weight %s" %(weight_path))
    pretrained = torch.load(weight_path) 
    pretrained_dict = pretrained['model_state_dict']
    model_dict = model.state_dict()
    for k, v in pretrained_dict.items():
        if k in model_dict: 
            model_dict[k].copy_(v)

pre_trained_weight = 'VGG_convnetckpt.pth'

load_pretrained_weight(net, pre_trained_weight)
eval_on_test_set() 
print(' ')
