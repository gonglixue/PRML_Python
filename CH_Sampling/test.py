import torch
import numpy as np
from torch.autograd import Variable

def test():
    x = torch.ones(1, 2)
    Sigma = torch.FloatTensor([[1, 0.8], [0.8, 1]])

    z = torch.ones(x.size())
    y = torch.matmul(x, Sigma)
    y = torch.matmul(y, x.t())
    print(y)

def test2():
    x = torch.ones(1, 2)
    x = Variable(x)
    y = torch.ones(1, 2)

    z = x + 0.5
    print(x.data)

def test3():
    v = torch.ones(1, 2)
    v = Variable(v).data
    temp = v**2
    temp = torch.sum(temp)
    print(temp)

def test4():
    High_D = 3
    High_cov = np.random.rand(High_D, High_D)
    High_cov = (High_cov + High_cov.T) / 2
    # High_cov = np.zeros(shape=(High_D, High_D))
    High_cov[np.arange(High_D), np.arange(High_D)] = 1.0
    # High_Sigma = torch.from_numpy(High_cov).float()
    mean = np.zeros(High_D)
    np_inital_vel = np.random.multivariate_normal(mean, High_cov, 1)

    print(np_inital_vel)

if __name__ == '__main__':
    test4()