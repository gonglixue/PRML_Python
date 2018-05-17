import torch
from torch.autograd import Variable

def scalar_grad():
    x = Variable(torch.ones(1)*3, requires_grad=True)   # 3
    y = Variable(torch.ones(1)*4, requires_grad=True)   # 4
    z = x.pow(2) + 3*y.pow(2)

    z.backward()

    # dz/dx = 2x
    # dz/dy = 6y

    print(x.grad)
    print(y.grad)

def vector_grad():
    x = Variable(torch.ones(2)*3, requires_grad=True)
    y = Variable(torch.ones(2)*4, requires_grad=True)
    z = x.pow(2) + 3*y.pow(2)
    z.backward(torch.ones(2))
    print(x.grad)
    print(y.grad)

def grad():
    W = Variable(torch.FloatTensor([[1, 1, 1], [2, 2, 2]]), requires_grad=True)
    x = Variable(torch.FloatTensor([1, 2, 3]), requires_grad=False)
    B = Variable(torch.FloatTensor([2, 2]), requires_grad=True)

    y = W.mv(x) + B.pow(2)
    y.backward(torch.ones(1))   # 

    print(W.grad)
    print(B.grad)

if __name__ == '__main__':
    grad()