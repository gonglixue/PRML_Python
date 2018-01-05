import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

from models.BasicModels import LogisticRegression

# hyper parameters
input_size = 28*28
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# MNIST Dataset
train_dataset = dsets.MNIST(root='/home/gonglixue/dataset/pytorch_datasets',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root='/home/gonglixue/dataset/pytorch_datasets',
                           train=False,
                           transform=transforms.ToTensor())

# Dataset Loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

model = LogisticRegression(input_size, num_classes)
model = model.cuda()

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Train
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images.view(-1, 28*28)).cuda()  # view()
        lables = Variable(labels).cuda()

        # forward
        out = model(images)
        loss = criterion(out, lables)

        # backward
        optimizer.zero_grad()  # ?
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print('Epoch:[%d/%d], Step:[%d/%d], Loss:%.4f'
                  % (epoch+1, num_epochs, i+1, len(train_loader)//batch_size,
                    loss.data[0]))

# test
