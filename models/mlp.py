### 1layer

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP1_layer(nn.Module):
  '''
    Multilayer Perceptron.
  '''
  def __init__(self, n_units, n_channels, n_classes=10):
    super().__init__()

    self.layers = nn.Sequential(
      # nn.Flatten(),
      nn.Linear(32 * 32 * n_channels, n_units),
      nn.Dropout(p=0.5),
      nn.ReLU(),
      nn.Linear(n_units, n_classes)
    )

  def forward(self, x):
    '''Forward pass'''
    return self.layers(x)


class MLP2_layer(nn.Module):
  '''
    Multilayer Perceptron.
  '''
  def __init__(self, n_units, n_channels, n_classes=10):
    super().__init__()

    self.layers = nn.Sequential(
      # nn.Flatten(),
      nn.Linear(32 * 32 * n_channels, n_units),
      nn.ReLU(),
      nn.Linear(n_units, n_units),
      nn.ReLU(),
      nn.Linear(n_units, n_classes)
    )

  def forward(self, x):
    '''Forward pass'''
    return self.layers(x)



class MLP4_layer(nn.Module):
  '''
    Multilayer Perceptron.
  '''
  def __init__(self, n_units, n_channels, n_classes=10):
    super().__init__()

    self.layers = nn.Sequential(
      # nn.Flatten(),
      nn.Linear(32 * 32 * n_channels, n_units),
      nn.ReLU(),
      nn.Linear(n_units, n_units),
      nn.ReLU(),
      nn.Linear(n_units, n_units),
      nn.ReLU(),
      nn.Linear(n_units, n_units),
      nn.ReLU(),
      nn.Linear(n_units, n_classes)
    )

  def forward(self, x):
    '''Forward pass'''
    return self.layers(x)


class MLP8_layer(nn.Module):
  '''
    Multilayer Perceptron.
  '''
  def __init__(self, n_units, n_channels, n_classes=10):
    super().__init__()

    self.layers = nn.Sequential(
      # nn.Flatten(),
      nn.Linear(32 * 32 * n_channels, n_units),
      nn.ReLU(),
      nn.Linear(n_units, n_units),
      nn.ReLU(),
      nn.Linear(n_units, n_units),
      nn.ReLU(),
      nn.Linear(n_units, n_units),
      nn.ReLU(),
      nn.Linear(n_units, n_units),
      nn.ReLU(),
      nn.Linear(n_units, n_units),
      nn.ReLU(),
      nn.Linear(n_units, n_units),
      nn.ReLU(),
      nn.Linear(n_units, n_units),
      nn.ReLU(),
      nn.Linear(n_units, n_classes)
    )

  def forward(self, x):
    '''Forward pass'''
    return self.layers(x)

class MLP16_layer(nn.Module):
  '''
    Multilayer Perceptron.
  '''
  def __init__(self, n_units, n_channels, n_classes=10):
    super().__init__()

    self.layers = nn.Sequential(
      # nn.Flatten(),
      nn.Linear(32 * 32 * n_channels, n_units),
      nn.ReLU(),
      nn.Linear(n_units, n_units),
      nn.ReLU(),
      nn.Linear(n_units, n_units),
      nn.ReLU(),
      nn.Linear(n_units, n_units),
      nn.ReLU(),
      nn.Linear(n_units, n_units),
      nn.ReLU(),
      nn.Linear(n_units, n_units),
      nn.ReLU(),
      nn.Linear(n_units, n_units),
      nn.ReLU(),
      nn.Linear(n_units, n_units),
      nn.ReLU(),
      nn.Linear(n_units, n_units),
      nn.ReLU(),
      nn.Linear(n_units, n_units),
      nn.ReLU(),
      nn.Linear(n_units, n_units),
      nn.ReLU(),
      nn.Linear(n_units, n_units),
      nn.ReLU(),
      nn.Linear(n_units, n_units),
      nn.ReLU(),
      nn.Linear(n_units, n_units),
      nn.ReLU(),
      nn.Linear(n_units, n_units),
      nn.ReLU(),
      nn.Linear(n_units, n_units),
      nn.ReLU(),
      nn.Linear(n_units, n_classes)
    )

  def forward(self, x):
    '''Forward pass'''
    return self.layers(x)


def MLP1Layer():
    return MLP1_layer(n_units=1024, n_channels=3, n_classes=10)

def MLP2Layer():
    return MLP2_layer(n_units=1024, n_channels=3, n_classes=10)

def MLP4Layer():
    return MLP4_layer(n_units=1024, n_channels=3, n_classes=10)

def MLP8Layer():
    return MLP8_layer(n_units=1024, n_channels=3, n_classes=10)

def MLP16Layer():
    return MLP16_layer(n_units=1024, n_channels=3, n_classes=10)

