import torch
import copy
from torch.autograd import Variable

def f(seed, model, sigma, environment, reward,safe_mutation):
    copyModel = copy.deepcopy(model)
    copyEnviroment = copy.deepcopy(environment)

    #S_SUM
    if safe_mutation==1:
        output_sum = Variable(torch.FloatTensor(10).zero_(), requires_grad=True)
        output_gradients = []
        for batch_idx, (data, target) in enumerate(copyEnviroment):
            batch_size = data.size()[0]
            data, target = Variable(data).resize(batch_size, 28*28), Variable(target)
            output = copyModel(data)
            for j in range(batch_size):
                output_sum.data += output.data[j, :]
        for k in range(10):
            output[1, k].data += (output_sum[k].data - output[1, k].data)
            copyModel.zero_grad()
            output[1, k].backward(retain_graph=True)
            output_gradient = []
            for parameter in copyModel.parameters():
                output_gradient.append(copy.deepcopy(parameter.grad))
            output_gradients.append(output_gradient)
        copyModel.zero_grad()
        for k in range(10):
            count = 0
            for parameter in copyModel.parameters():
                parameter.grad += output_gradients[k][count] ** 2
                count += 1
        for parameter in copyModel.parameters():
            parameter.grad = parameter.grad ** 0.5

    elif safe_mutation==2:
        print("safe mutattion 2")

    elif safe_mutation==3:
        print("safe mutattion 3")
    else:
        for parameter in copyModel.parameters():
            torch.manual_seed(seed)
            if len(list(parameter.size())) == 1:
                s = torch.Tensor(list(parameter.size())[0])
                torch.nn.init.normal(s, 0, sigma)
            else:
                s = torch.Tensor(list(parameter.size())[0], list(parameter.size())[1])
                torch.nn.init.normal(s, 0, sigma)
            parameter.data += s
        value = 0
        for batch_idx, (data, target) in enumerate(copyEnviroment):
            data, target = Variable(data).resize(data.size()[0], 28 * 28), Variable(target)
            output = copyModel(data)
            value += reward(output, target).data.numpy()
        return -value[0]



    for parameter in copyModel.parameters():
        torch.manual_seed(seed)
        if len(list(parameter.size())) == 1:
            s = torch.Tensor(list(parameter.size())[0])
            torch.nn.init.normal(s, 0, sigma)
        else:
            s = torch.Tensor(list(parameter.size())[0], list(parameter.size())[1])
            torch.nn.init.normal(s, 0, sigma)
        parameter.data += s / parameter.grad.data
    value = 0
    for batch_idx, (data, target) in enumerate(copyEnviroment):
        data, target = Variable(data).resize(data.size()[0], 28 * 28), Variable(target)
        output = copyModel(data)
        value += reward(output, target).data.numpy()
    return -value[0]