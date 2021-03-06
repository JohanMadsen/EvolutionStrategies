#Importing Liberies
##################################################
import torch
from torch.multiprocessing import Pool
import random
from functools import partial
from torch.autograd import Variable
from torchvision import datasets, transforms
from scipy.stats import rankdata
import numpy as np
import argparse
import copy
import math


#SETUP MNIST
##################################################
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
kwargs={}

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)





#The paralized function, that takes a mutations, applies that to the NN and then calculates a fitness for that mutation
##################################################
def f(mutation, data,target):
    reward = torch.nn.CrossEntropyLoss()
    copyModel = copy.deepcopy(model)
    count=0
    for parameter in copyModel.parameters():
        s = mutation[count]
        parameter.data += s
        count +=1
    output = copyModel(data)
    value = reward(output, target).data.numpy()
    return -value[0]


#Settings
##################################################
safe_mutation=0
adam=1
batchnorm=0
random.seed(1000)
torch.manual_seed(1000)
num_processes = 1
n = 120
num_epoch=50
sigma = 1e-4
learning_rate = 1
reward = torch.nn.CrossEntropyLoss()

#Model
##################################################
N, D_in, H, D_out = 1000, 28*28, 100, 10

if batchnorm==1:
    model = torch.nn.Sequential(
        torch.nn.BatchNorm1d(D_in),
        torch.nn.Linear(D_in, H),
        torch.nn.BatchNorm1d(H),
        torch.nn.ReLU(),
        torch.nn.BatchNorm1d(H),
        torch.nn.Linear(H, D_out),
    )
if batchnorm==2:
    model = torch.nn.Sequential(
        torch.nn.BatchNorm1d(D_in),
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out),
    )
else:
    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out),
    )

model.share_memory()


if adam==1:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#The main function
##################################################
if __name__ == "__main__":
    with Pool(num_processes) as p:
        for t in range(num_epoch+1): # After each Epoc write out how good the network is on test data
            if t%1==0:
                value=0
                value2=0
                sum=0
                for batch_idx, (data, target) in enumerate(test_loader):
                    data, target = Variable(data).resize(data.size()[0],D_in), Variable(target)
                    output = model(data)
                    values,indices=output.max(1)
                    value2+=reward(output,target)
                    sum+=target.data.size()[0]
                    for i in range(target.data.size()[0]):
                        if(indices.data[i]!=target.data[i]):
                            value+=1
                print("On Test data:","Epoch:",t,"Loss:",value2.data.numpy(),"Procent wrong:",value/sum)
            if t==num_epoch:
                break
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = Variable(data).resize(data.size()[0], D_in), Variable(target)
                output = model(data)
                score=reward(output,target).data.numpy()
                lastit=-score[0]
                seeds = random.sample(range(1000000), n)
                mutations=[]
                if safe_mutation==1:# Calculates the safe mutation S_SUM
                    output_sum = Variable(torch.FloatTensor(10).zero_(), requires_grad=True)
                    output_gradients = []
                    scale = []
                    batch_size = data.size()[0]
                    for j in range(batch_size):
                        output_sum.data += output.data[j, :]
                    for k in range(10):
                        output[1, k].data += (output_sum[k].data - output[1, k].data)
                        model.zero_grad()
                        output[1, k].backward(retain_graph=True)
                        output_gradient = []
                        for parameter in model.parameters():
                            output_gradient.append(parameter.grad)
                        output_gradients.append(output_gradient)
                elif safe_mutation==2:# Calculates the safe mutation S_ABS
                    output_gradients = []
                    scale = []
                    batch_size = data.size()[0]
                    for i in range(10):
                        output_gradient = []
                        for j in range(batch_size):
                            model.zero_grad()
                            output[j, i].backward(retain_graph=True)
                            if j==0:
                                for parameter in model.parameters():
                                    output_gradient.append(parameter.grad.abs())
                            else:
                                count=0
                                for parameter in model.parameters():
                                    output_gradient[count]+=parameter.grad.abs()
                                    count+=1
                        count=0
                        for parameter in model.parameters():
                            output_gradient[count]=output_gradient[count]/batch_size
                            count+=1
                        output_gradients.append(output_gradient)
                if safe_mutation>0:#Samples the mutations and applies the weight given by the "Safe mutation"
                    for k in range(10):
                        if k == 0:
                            count=0
                            for parameter in model.parameters():
                                scale.append(output_gradients[k][count] ** 2)
                                count +=1
                        else:
                            count = 0
                            for parameter in model.parameters():
                                scale[count] += output_gradients[k][count] ** 2
                                count +=1
                    for i in range(len(scale)):
                        scale[i] = scale[i] ** 0.5
                        scale[i] *= (scale[i]>1e-1).type(torch.FloatTensor)
                        scale[i] += (scale[i] == 0).type(torch.FloatTensor)
                    for i in range(n):
                        mutation = []
                        torch.manual_seed(seeds[i])
                        count=0
                        for parameter in model.parameters():
                            if len(list(parameter.size())) == 1:
                                s = torch.Tensor(list(parameter.size())[0])
                            else:
                                s = torch.Tensor(list(parameter.size())[0], list(parameter.size())[1])
                            torch.nn.init.normal(s, 0, sigma)
                            s=s/scale[count].data
                            mutation.append(s)
                            count+=1
                        mutations.append(mutation)
                else:#Samples the mutations without "Safe Mutations"
                    for i in range(n):
                        mutation = []
                        torch.manual_seed(seeds[i])
                        for parameter in model.parameters():
                            if len(list(parameter.size())) == 1:
                                s = torch.Tensor(list(parameter.size())[0])
                            else:
                                s = torch.Tensor(list(parameter.size())[0], list(parameter.size())[1])
                            torch.nn.init.normal(s, 0, sigma)
                            mutation.append(s)
                        mutations.append(mutation)

                par_f = partial(f, data=data, target=target) #Makes a partial function so each mutation can be run on the same data
                result = p.map(par_f, mutations) # Calls the paralized function with each mutation
                #Calculates the rank transform of the fitness values
                f_values = rankdata(result)
                number=np.sum(result >= lastit)
                if number == 0 or number == n:
                    number = n / 2
                #number=n/2
                for i in range(len(f_values)):
                    f_values[i] *= -1
                    f_values[i] += n + 1
                f_rank_values = [0] * n
                count=0
                fsum=0
                for value in f_values:
                    fsum+=max(0.0,math.log(number+1)-math.log(value))
                for value in f_values:
                    f_rank_values[count]=(max(0.0,math.log(number+1)-math.log(value))/fsum)-1/n
                    count+=1

                #Estimates the gradiant from the fitness values and the mutations, and applies it to take one step
                count=0
                for parameter in model.parameters():
                    s=0
                    for i in range(n):
                        s1=mutations[i][count]
                        s += s1 * f_rank_values[i]
                    count +=1
                    #Normal gradient descent
                    if adam==0:
                        parameter.data += (learning_rate/(n*sigma))*s
                    #ADAM optimizer
                    if adam==1:
                        parameter.grad=-(1/(n*sigma))*Variable(s)
                if adam==1:
                    optimizer.step()
