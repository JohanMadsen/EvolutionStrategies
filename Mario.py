#Importing Liberies
##################################################
from torch.multiprocessing import Pool
import copy
import random
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import math
from PIL import Image
from scipy.stats import rankdata
from torch.autograd import Variable
import gym


FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor
ByteTensor = torch.ByteTensor
Tensor = FloatTensor



#Model
##################################################
class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.head = nn.Linear(640, 12)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


resize = T.Compose([T.ToPILImage(),
                   T.Resize((56,64), interpolation=Image.CUBIC),
                    T.ToTensor()])


#This is  to transform the pictures into input for the NN
def get_screen(obs):
    s=np.ascontiguousarray(obs, dtype=np.float32) / 255
    s = torch.from_numpy(s)
    return resize(s).unsqueeze(0).type(Tensor)




#Settings
##################################################
safe_mutation=0
adam=0
random.seed(1000)
torch.manual_seed(1000)
num_processes = 1
n = 120
sigma = 1e-2
learning_rate = 1e-0

model = DQN()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)




#The paralized function, that takes a mutations, applies that to the NN and then calculates a fitness for that mutation
##################################################
def f(mutation):
    env = gym.make('SuperMarioBros-1-1-v0')
    copyModel = copy.deepcopy(model)
    count=0
    for parameter in copyModel.parameters():
        s = mutation[count]
        parameter.data += s
        count +=1
    observation = torch.from_numpy(env.reset())
    last_screen = get_screen(observation)
    current_screen = last_screen
    state = current_screen - last_screen
    action = np.random.randint(2, size=6)
    done = False
    notmoving=0
    while not done:
        result = copyModel(Variable(state, requires_grad=True).type(FloatTensor))
        for i in range(6):
            if result[0, i * 2].data.numpy()[0] > result[0, i * 2 + 1].data.numpy()[0]:
                action[i] = 1
            else:
                action[i] = 0

        observation, reward, done, info = env.step(action)  # feedback from environment
        last_screen = current_screen
        current_screen = get_screen(observation)
        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None
        state = next_state
        if state is None:
            break
        if info.get("time")<390 and info.get("distance")<50:
            break
        if info.get("time")<5:
            break
        if torch.sum(state)==0.0:
            notmoving+=1
        else:
            notmoving=0
        if notmoving>100:
            break
    env.close()
    return info.get("distance")

# A function to do a rollout of the enviroment to see how good this iteration of the NN is
def rollout():
    env = gym.make('SuperMarioBros-1-1-v0')
    observation = torch.from_numpy(env.reset())
    last_screen = get_screen(observation)
    current_screen = last_screen
    state = current_screen - last_screen
    action = np.random.randint(2, size=6)
    done = False
    notmoving=0
    while not done:
        result = model(Variable(state, requires_grad=True).type(FloatTensor))
        for i in range(6):
            if result[0, i * 2].data.numpy()[0] > result[0, i * 2 + 1].data.numpy()[0]:
                action[i] = 1
            else:
                action[i] = 0

        observation, reward, done, info = env.step(action)  # feedback from environment
        last_screen = current_screen
        current_screen = get_screen(observation)
        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None
        state = next_state
        if state is None:
            break
        if info.get("time")<390 and info.get("distance")<50:
            break
        if info.get("time")<5:
            break
        if torch.sum(state)==0.0:
            notmoving+=1
        else:
            notmoving=0
        if notmoving>100:
            break
    print(info.get("distance"))
    env.close()



def es():#MY Evolutions strategies function
    seeds = random.sample(range(1000000), n)
    mutations = []
    if safe_mutation == 1:  # Calculates the safe mutation S_SUM
        env = gym.make('SuperMarioBros-1-1-v0')
        output_gradients = []
        observation = torch.from_numpy(env.reset())
        last_screen = get_screen(observation)
        current_screen = last_screen
        state = current_screen - last_screen
        action = np.random.randint(2, size=6)
        output_sum = Variable(torch.FloatTensor(12).zero_(), requires_grad=True)
        done = False
        notmoving = 0
        while not done:
            result = model(Variable(state, requires_grad=True).type(FloatTensor))
            output_sum.data += result.data
            for i in range(6):
                if result[0, i * 2].data.numpy()[0] > result[0, i * 2 + 1].data.numpy()[0]:
                    action[i] = 1
                else:
                    action[i] = 0

            observation, reward, done, info = env.step(action)  # feedback from environment
            last_screen = current_screen
            current_screen = get_screen(observation)
            if not done:
                next_state = current_screen - last_screen
            else:
                next_state = None
            state = next_state
            if state is None:
                break
            if info.get("time") < 390 and info.get("distance") < 50:
                break
            if info.get("time") < 5:
                break
            if torch.sum(state) == 0.0:
                notmoving += 1
            else:
                notmoving = 0
            if notmoving > 100:
                break
        env.close()
        result.data += (output_sum.data - result.data)
        for k in range(2):
            model.zero_grad()
            result[0,k].backward(retain_graph=True)
            output_gradient = []
            for parameter in model.parameters():
                output_gradient.append(parameter.grad)
            output_gradients.append(output_gradient)
    if safe_mutation > 0:#Samples the mutations and applies the weight given by the "Safe mutation"
        scale=[]
        for k in range(2):
            if k == 0:
                count = 0
                for parameter in model.parameters():
                    scale.append(output_gradients[k][count] ** 2)
                    count += 1
            else:
                count = 0
                for parameter in model.parameters():
                    scale[count] += output_gradients[k][count] ** 2
                    count += 1
        for i in range(len(scale)):
            scale[i] = scale[i] ** 0.5
            scale[i] *= (scale[i] > 1e-1).type(torch.FloatTensor)
            scale[i] += (scale[i] == 0).type(torch.FloatTensor)
        for i in range(n):
            mutation = []
            torch.manual_seed(seeds[i])
            count = 0
            for parameter in model.parameters():
                if len(list(parameter.size())) == 1:
                    s = torch.Tensor(list(parameter.size())[0])
                elif len(list(parameter.size())) == 2:
                    s = torch.Tensor(list(parameter.size())[0], list(parameter.size())[1])
                elif len(list(parameter.size())) == 3:
                    s = torch.Tensor(list(parameter.size())[0], list(parameter.size())[1], list(parameter.size())[2])
                else:
                    s = torch.Tensor(list(parameter.size())[0], list(parameter.size())[1], list(parameter.size())[2],
                                     list(parameter.size())[3])
                torch.nn.init.normal(s, 0, sigma)
                s = s / scale[count].data
                mutation.append(s)
                count += 1
            mutations.append(mutation)
    else:#Samples the mutations without "Safe Mutations"
        for i in range(n):
            mutation = []
            torch.manual_seed(seeds[i])
            for parameter in model.parameters():
                if len(list(parameter.size())) == 1:
                    s = torch.Tensor(list(parameter.size())[0])
                elif len(list(parameter.size())) == 2:
                    s = torch.Tensor(list(parameter.size())[0], list(parameter.size())[1])
                elif len(list(parameter.size())) == 3:
                    s = torch.Tensor(list(parameter.size())[0], list(parameter.size())[1], list(parameter.size())[2])
                else:
                    s = torch.Tensor(list(parameter.size())[0], list(parameter.size())[1], list(parameter.size())[2],
                                     list(parameter.size())[3])
                torch.nn.init.normal(s, 0, sigma)
                mutation.append(s)
            mutations.append(mutation)
    par_f = partial(f)
    result = p.map(par_f, mutations)# Calls the paralized function with each mutation

    # Calculates the rank transform of the fitness values
    f_values = rankdata(result)
    f_rank_values = [0] * n
    fsum = 0
    for i in range(len(f_values)):
        f_values[i]*=-1
        f_values[i]+=n+1
    count=0
    for value in f_values:
        fsum += max(0.0, math.log(n / 2 + 1) - math.log(value))
    for value in f_values:
        f_rank_values[count] = (max(0.0, math.log(n / 2 + 1) - math.log(value))/fsum) - 1 / n
        count += 1

    # Estimates the gradiant from the fitness values and the mutations, and applies it to take one step
    count = 0
    for parameter in model.parameters():
        s = 0
        for i in range(n):
            s1 = mutations[i][count]
            s += s1 * f_rank_values[i]
        count += 1
        # Normal gradient descent
        if adam == 0:
            parameter.data += (learning_rate / (n * sigma)) * s
        # ADAM optimizer
        if adam == 1:
            parameter.grad = -(1 / (n * sigma)) * Variable(s)
    if adam == 1:
        optimizer.step()
    rollout()

num_episodes = 1000
with Pool(num_processes) as p:
    rollout()
    for i_episode in range(num_episodes):
        print(i_episode+1)
        es()
print('Complete')
