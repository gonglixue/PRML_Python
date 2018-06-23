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

    u = Variable(torch.FloatTensor([0, 0, 0]), requires_grad=False)

    y = W.mv(x-u) + B.pow(2)
    z = W.mv(x - u) + B.pow(2)

    # y.backward(torch.ones(1))   #
    #
    # print(W.grad)
    # print(B.grad)
    #
    # W.grad.data.zero_()
    # B.grad.data.zero_()
    #
    # z.backward(torch.ones(1))
    #
    # print(W.grad)
    # print(B.grad)

    r = y + z
    r.backward(torch.ones(1))
    print(W.grad)
    print(B.grad)

def grad2():
    W = Variable(torch.rand(2, 2), requires_grad=True)
    W2 = Variable(torch.rand(2, 1), requires_grad=True)
    x1 = Variable(torch.rand(1, 2), requires_grad=True)
    x2 = Variable(torch.rand(1, 2), requires_grad=True)

    print("w: ")
    print(W)
    print("x1: ")
    print(x1)
    print("x2: ")
    print(x2)
    print("--------------------")

    y1 = torch.matmul(torch.matmul(x1, W), W2)
    print(torch.matmul(W, W2))
    # y = Variable(y, requires_grad=True)
    # print("y1:")
    # print(y1)

    y1.backward()
    # print(W.grad)
    print(x1.grad)

    # W.grad.data.zero_()
    # x1.grad.data.zero_()
    y2 = torch.matmul(torch.matmul(x2, W), W2)
    y2.backward()
    # print("y2: ")
    # print(y2)
    # print(W.grad)
    print(x2.grad)

def test_dimension():
    batch_size = 3
    dim = 2
    x = torch.rand(batch_size, dim)
    u = torch.zeros(dim)
    W = torch.rand(dim, dim)

    x = Variable(x, requires_grad=True)
    u = Variable(u, requires_grad=True)
    W = Variable(W, requires_grad=True)
    print(x)
    print(u)
    print(W)

    temp = x - u
    y = torch.matmul(torch.matmul(temp, W), torch.sum(temp.t(), dim=1, keepdim=True))
    y.backward(torch.ones(batch_size, dim))
    print('--------')
    # print(y)
    print(x.grad)
    print(torch.matmul(x, W))

if __name__ == '__main__':
    test_dimension()