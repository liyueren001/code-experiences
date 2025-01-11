import torch
import torch.nn as nn


# Define the LeNet model
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        
        """=============== Your code below ================="""
        # make sure your layers use the default dtype (aka torch.float32)
        
        # Conv1: 1 input channel, 6 output channels, 5x5 kernel size
        # Conv2: 6 input channels, 16 output channels, 5x5 kernel size
        # FC1: 16*4*4 input features, 120 output features
        # FC2: 120 input features, 84 output features
        # FC3: 84 input features, 10 output features
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5)
        self.conv2= nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5)
        self.fc1 = nn.Linear(in_features=16 * 4 * 4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)
        """=============== Your code above ================="""

    def forward(self, x):
        """=============== Your code below ================="""
        # Conv1 -> ReLU -> MaxPool2d(2x2) -> Conv2 -> ReLU -> MaxPool2d(2x2) -> Resize -> FC1 -> ReLU -> FC2 -> ReLU -> FC3
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.max_pool2d(x,kernel_size=2)
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.max_pool2d(x, kernel_size=2)
        x = x.view(-1, 16 * 4 * 4)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        """=============== Your code above ================="""
        return x


if __name__ == "__main__":
    from check import validate_model
    validate_model(LeNet)