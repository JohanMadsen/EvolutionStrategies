from torch.multiprocessing import Pool
import copy
import random
from collections import namedtuple
from functools import partial
from itertools import count
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import math
from PIL import Image
from scipy.stats import rankdata
from torch.autograd import Variable


env = gym.make('CartPole-v0').unwrapped

plt.ion()

FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor
ByteTensor = torch.ByteTensor
Tensor = FloatTensor


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.head = nn.Linear(448, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


resize = T.Compose([T.ToPILImage(),
                    T.Scale(40, interpolation=Image.CUBIC),
                    T.ToTensor()])

# This is based on the code from gym.
screen_width = 600

def get_cart_location(env):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART


def get_screen(env):
    screen = env.render(mode='rgb_array').transpose(
        (2, 0, 1))
    # Strip off the top and bottom of the screen
    screen = screen[:, 160:320]
    view_width = 320
    cart_location = get_cart_location(env)
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    # Strip off the edges, so that we have a square image centered on a cart
    screen = screen[:, :, slice_range]
    # Convert to float, rescare, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0).type(Tensor)





#MODEL TRAINING and other settings
safe_mutation=1
adam=1
random.seed(1000)
torch.manual_seed(1000)
num_processes = 4
n = 120
sigma = 1e-1#1e-1 for SM
learning_rate = 1e-1#1e-1 for adam

GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

model = DQN()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

episode_durations = []


def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.FloatTensor(episode_durations)
    plt.title('Training...')
    plt.xlabel('Iteration')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    if len(durations_t) >= 5:
        means = durations_t.unfold(0, 5, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(4), means))
        plt.plot(means.numpy())
    plt.pause(0.001)  # pause a bit so that plots are updated


######################################################################
def f(mutation):
    env = gym.make('CartPole-v0').unwrapped
    copyModel = copy.deepcopy(model)
    count=0
    for parameter in copyModel.parameters():
        s = mutation[count]
        parameter.data += s
        count +=1
    totalrounds = 0
    for i in range(5):
        env.reset()
        last_screen = get_screen(env)
        current_screen = get_screen(env)
        state = current_screen - last_screen

        done=False
        while(not done):
            # Select and perform an action
            action = copyModel(
                Variable(state, volatile=True).type(FloatTensor)).data.max(1)[1].view(1, 1)
            _, reward, done, _ = env.step(action[0, 0])
            #Observe new state
            last_screen = current_screen
            current_screen = get_screen(env)
            if not done:
                next_state = current_screen - last_screen
            else:
                next_state = None
            # Move to the next state
            state = next_state
            totalrounds+=1
            if done:
                if i<4:
                    break
                else:
                    env.close()
                    plt.close()
                    return totalrounds

def rollout():
    w=0
    z=20
    for i in range(z):
        env.reset()
        last_screen = get_screen(env)
        current_screen = get_screen(env)
        state = current_screen - last_screen
        actions=list()
        while True:
            # Select and perform an action
            result=model(
                Variable(state, volatile=True).type(FloatTensor))
            action = result.data.max(1)[1].view(1, 1)
            actions.append(result)
            _, reward, done, _ = env.step(action[0, 0])
            # Observe new state
            last_screen = current_screen
            current_screen = get_screen(env)
            if not done:
                next_state = current_screen - last_screen
            else:
                next_state = None
            # Move to the next state
            state = next_state
            w+=1
            if done:
                if i==(z-1):
                    episode_durations.append(w/z)
                    plot_durations()
                break


def es():#MY Evolutions strategies function
    rollout()
    seeds = random.sample(range(1000000), n)
    mutations = []
    if safe_mutation == 1:  # SUM
        output_gradients = []
        w = 0
        ##rollout #####
        for i in range(5):
            env.reset()
            last_screen = get_screen(env)
            current_screen = get_screen(env)
            state = current_screen - last_screen
            output_sum = Variable(torch.FloatTensor(2).zero_(), requires_grad=True)
            while True:
                # Select and perform an action
                result = model(
                    Variable(state, volatile=True).type(FloatTensor))
                result1 = model(
                    Variable(state, requires_grad=True).type(FloatTensor))

                action = result.data.max(1)[1].view(1, 1)
                output_sum.data += result1.data
                _, reward, done, _ = env.step(action[0, 0])
                # Observe new state
                last_screen = current_screen
                current_screen = get_screen(env)
                if not done:
                    next_state = current_screen - last_screen
                else:
                    next_state = None
                # Move to the next state
                state = next_state
                w+=1
                if done:
                    break
        result1.data += (output_sum.data - result1.data)
        for k in range(2):
            model.zero_grad()
            result1[0,k].backward(retain_graph=True)
            output_gradient = []
            for parameter in model.parameters():
                output_gradient.append(parameter.grad)
            output_gradients.append(output_gradient)



    if safe_mutation > 0:
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
    else:
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
    result = p.map(par_f, mutations)
    f_values = rankdata(result)
    print(result)
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




last_sync = 0

num_episodes = 1000
with Pool(num_processes) as p:
    for i_episode in range(num_episodes):
        es()
print('Complete')
env.render(close=True)
env.close()
plt.ioff()
plt.show()
