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

if __name__ == '__main__':
    test3()