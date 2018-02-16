import torch
import copy
from torch.autograd import Variable

def f(seed, model, sigma, environment, reward):
    copyModel = copy.deepcopy(model)
    copyEnviroment = copy.deepcopy(environment)
    for parameter in copyModel.parameters():
        torch.manual_seed(seed)
        if len(list(parameter.size())) == 1:
            s = torch.Tensor(list(parameter.size())[0])
            torch.nn.init.normal(s,0,sigma)
        else:
            s = torch.Tensor(list(parameter.size())[0], list(parameter.size())[1])
            torch.nn.init.normal(s,0,sigma)
        parameter.data += s
    value = 0
    for batch_idx, (data, target) in enumerate(copyEnviroment):
        data, target = Variable(data).resize(data.size()[0], 28*28), Variable(target)
        output = copyModel(data)
        value+=reward(output,target).data.numpy()
    return -value[0]