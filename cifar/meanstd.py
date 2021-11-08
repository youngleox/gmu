import numpy as np
# load data
from torchvision import datasets

# load the training data
train_data = datasets.CIFAR10('./cifar10_data', train=True, download=True)
# use np.concatenate to stick all the images together to form a 1600000 X 32 X 3 array
x = np.concatenate([np.asarray(train_data[i][0]) for i in range(len(train_data))])
# print(x)
print(x.shape)
# calculate the mean and std along the (0, 1) axes
train_mean = np.mean(x, axis=(0, 1))/255
train_std = np.std(x, axis=(0, 1))/255
# the the mean and std
print(train_mean, train_std)