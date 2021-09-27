import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np
from utils import train_one_shot, compute_accuracy, visualize
from model import Lenet_300_100

np.random.seed(42)
epochs = 15

trans = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])

root = './data'
if not os.path.exists(root):
    os.mkdir(root)
train_set = datasets.MNIST(root=root, train=True,
                           transform=trans, download=True)
test_set = datasets.MNIST(root=root, train=False,
                          transform=trans, download=True)

batch_size = 60
train_loader = torch.utils.data.DataLoader(
    dataset=train_set,
    batch_size=batch_size,
    shuffle=True)

test_loader = torch.utils.data.DataLoader(
    dataset=test_set,
    batch_size=batch_size,
    shuffle=False)

net = Lenet_300_100()

if torch.cuda.is_available():
    print('CUDA enabled.')
    net.cuda()

# Retrain
loss_func = nn.CrossEntropyLoss()  # LogSoftmax + NLLLoss in single class.
optimizer = torch.optim.SGD(net.parameters(), lr=1.2e-3,
                            weight_decay=0.0001)

history = train_one_shot(net, loss_func, optimizer,
                         prepare_epochs=epochs, prune_epochs=epochs,
                         prune_percent=75, train_loader=train_loader,
                         val_loader=test_loader)

print("--- After retraining ---")
print("Test accuracy: {}".format(compute_accuracy(net, test_loader)))

plt.figure(figsize=(15, 6))
for i, key in enumerate(['train_loss', 'validation_accuracy']):
    plt.subplot(1, 2, i+1)
    for stage in ['train-pre-prune', 'train-post-prune']:
        plt.plot(history[stage][key], label=stage)
    plt.ylabel(key)
    plt.xlabel('epoch')
    plt.legend()
plt.show()

visualize(net.fc300.weight, 'FC weight', (8, 8))
