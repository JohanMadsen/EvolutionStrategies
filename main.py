import torch
from torch.multiprocessing import Pool
import random
from functools import partial
from torch.autograd import Variable
from torchvision import datasets, transforms
from scipy.stats import rankdata
import argparse
import es as es
import time


#SETUP MNIST
##################################################
# Training settings
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



torch.manual_seed(1000)

N, D_in, H, D_out = 64, 1*28*28, 100, 10
model = torch.nn.Sequential(
    torch.nn.BatchNorm1d(D_in),
    torch.nn.Linear(D_in, H),
    torch.nn.BatchNorm1d(H),
    torch.nn.ReLU(),
    torch.nn.BatchNorm1d(H),
    torch.nn.Linear(H, D_out),
)

random.seed(1000)
num_processes = 4
n = 8
sigma=1e-3
learning_rate = 1e-3
reward=torch.nn.CrossEntropyLoss()
if __name__ == "__main__":
    with Pool(num_processes) as p:
        for t in range(100):
            if t%1==0:
                value=0
                sum=0
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = Variable(data).resize(data.size()[0],D_in), Variable(target)
                    output = model(data)
                    values,indices=output.max(1)
                    sum+=target.data.size()[0]
                    for i in range(target.data.size()[0]):
                        if(indices.data[i]!=target.data[i]):
                            value+=1
                print("Iteration:",t,"Loss:",value/sum)
            seeds = random.sample(range(1000000), n)
            par_f = partial(es.f, model=model, sigma=sigma, environment=train_loader, reward=reward)
            result = p.map(par_f, seeds)
            f_values = rankdata(result)
            f_rank_values = [0] * n
            count=0
            for value in f_values:
                f_rank_values[count]=value-n/2
                if f_rank_values[count]<=0:
                    f_rank_values[count]=-n/4+3/2
                count=count+1
            for parameter in model.parameters():
                for i in range(n):
                    torch.manual_seed(seeds[i])
                    if i == 0:
                        if len(list(parameter.size())) == 1:
                            s = torch.Tensor(list(parameter.size())[0])
                            s1 = torch.Tensor(list(parameter.size())[0])
                            torch.nn.init.normal(s, 0, sigma)
                            s *= f_rank_values[i]
                        else:
                            s = torch.Tensor(list(parameter.size())[0], list(parameter.size())[1])
                            s1 = torch.Tensor(list(parameter.size())[0], list(parameter.size())[1])
                            torch.nn.init.normal(s, 0, sigma)
                            s *= f_rank_values[i]
                    else:
                        torch.nn.init.normal(s1, 0, sigma)
                        s += s1*f_rank_values[i]
                parameter.data += (learning_rate/(n*sigma))*s


