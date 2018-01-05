import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable

from models.BasicModels import LinearRegression

# Hyper Parameters
input_size = 1
output_size = 1
num_epochs = 60
learning_rate = 0.001

x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

linear_model = LinearRegression(input_size, output_size)

# Loss and Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(linear_model.parameters(), lr=learning_rate)

# Train
for epoch in range(num_epochs):
    inputs = Variable(torch.from_numpy(x_train))
    targets = Variable(torch.from_numpy(y_train))

    optimizer.zero_grad()
    outputs = linear_model(inputs)
    loss = criterion(outputs, targets)  # loss type: Variable
    loss.backward()
    optimizer.step()

    if (epoch+1)%5 == 0:
        print('Epoch [%d/%d], Loss:%.4f'
              % (epoch+1, num_epochs, loss.data[0]))

# Save the Model
torch.save(linear_model.state_dict(), 'check_points/linear_model.pkl')

# plot
predicted = linear_model(Variable(torch.from_numpy(x_train))).data.numpy()
plt.plot(x_train, y_train, 'ro', label='Original data')
plt.plot(x_train, predicted, label='Fitted line')
plt.legend()
plt.show()

