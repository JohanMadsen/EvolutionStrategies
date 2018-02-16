random.seed(1000)
num_processes = 8
n = 100
sigma=1e-3
learning_rate = 1e-3
env=list()
env.append(x)
env.append(y)
reward=torch.nn.MSELoss(size_average=False)

with Pool(num_processes) as p:
    for t in range(1):
        #t1 = time.time()
        if t%10==0:
             y_pred = model(x)
             loss = reward(y_pred, y)
             print(t, loss)
        seeds = random.sample(range(1000000), n)
        #t2 = time.time()
        par_f = partial(es.f, model=model, sigma=sigma, environment=env, reward=reward)
        result = p.map_async(par_f, seeds)
        result.wait()
        f_values = rankdata(result.get())
        f_rank_values=[0]*n
        #t3=time.time()
        count=0
        for value in f_values:
            f_rank_values[count]=value-n/2
            if f_rank_values[count]<=0:
                f_rank_values[count]=-1
            count=count+1
        for parameter in model.parameters():
            for i in range(n):
                torch.manual_seed(seeds[i])
                scale=f_rank_values[i]*learning_rate/n
                if i == 0:
                    if len(list(parameter.size())) == 1:
                        s = torch.randn(list(parameter.size())[0]) * scale
                    else:
                        s = torch.randn(list(parameter.size())[0], list(parameter.size())[1]) * scale
                else:
                    if len(list(parameter.size())) == 1:
                        s += torch.randn(list(parameter.size())[0]) * scale
                    else:
                        s += torch.randn(list(parameter.size())[0], list(parameter.size())[1]) * scale
            parameter.data += s
        #t4=time.time()
        #print((t4 - t3) * 100 / (t4 - t1))

    y_pred = model(x)
    loss = reward(y_pred, y)
    print(loss)













import torch
import copy


def f(seed, model, sigma, environment, reward):
    #t1=time.time()
    copyModel = copy.deepcopy(model)
    #t2=time.time()
    torch.manual_seed(seed)
    for parameter in copyModel.parameters():
        if len(list(parameter.size())) == 1:
            s = torch.randn(list(parameter.size())[0])*sigma
        else:
            s = torch.randn(list(parameter.size())[0], list(parameter.size())[1])*sigma
        parameter.data += s#parameter.data.add(1, s)
    #t3=time.time()
    y_pred = copyModel(environment[0])
    y = environment[1]
    value = reward(y_pred, y).data.numpy()
    #t4=time.time()
    #print((t3-t2)*100/(t4-t1))
    return -value[0]