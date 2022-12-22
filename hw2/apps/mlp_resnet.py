import sys
sys.path.append('../python')
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)

def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    """
    Args:
        dim (*int*): input dim
        hidden_dim (*int*: hidden dim
        norm (*nn.Module*): normalization method
        drop_prob (*float*): dropout probability
    """
    return nn.Sequential(nn.Residual(nn.Sequential(nn.Linear(dim, hidden_dim), norm(hidden_dim), nn.ReLU(), nn.Dropout(drop_prob), nn.Linear(hidden_dim, dim), norm(dim))), nn.ReLU())


def MLPResNet(dim, hidden_dim=100, num_blocks=3, num_classes=10, norm=nn.BatchNorm1d, drop_prob=0.1):
    """
    Args:
        dim (*int*): input dim
        hidden_dim (*int*): hidden dim
        num_blocks (*int*): number of ResidualBlocks
        num_classes (*int*): number of classes
        norm (*nn.Module*): normalization method
        drop_prob (*float*): dropout probability (0.1)
    """
    return nn.Sequential(nn.Linear(dim, hidden_dim), nn.ReLU(), *[ResidualBlock(hidden_dim, hidden_dim//2, norm, drop_prob) for _ in range(num_blocks)], nn.Linear(hidden_dim, num_classes))




def epoch(dataloader, model, opt=None):
    """
    Args:
        dataloader (*`needle.data.DataLoader`*): dataloader returning samples from the training dataset
        model (*`needle.nn.Module`*): neural network
        opt (*`needle.optim.Optimizer`*): optimizer instance, or `None`
    """
    if opt is None:
        model.eval()
    else:
        model.train()
    losses = []
    accuracyes = []
    np.random.seed(4)
    for batch in dataloader:
        batch_x, batch_y = batch
        logits = model(batch_x.reshape((batch_x.shape[0], 784)))
        loss = nn.SoftmaxLoss()(logits, batch_y)
        if opt is not None:
            opt.reset_grad()
            loss.backward()
            opt.step()
        losses.append(loss.numpy())
        accuracyes.append((np.argmax(logits.numpy(), axis=1) == batch_y.numpy()).mean())
    return 1 - np.mean(accuracyes), np.mean(losses)

def train_mnist(batch_size=100, epochs=10, optimizer=ndl.optim.Adam,
                lr=0.001, weight_decay=0.001, hidden_dim=100, data_dir="data"):
    """
    Args:
        batch_size (*int*): batch size to use for train and test dataloader
        epochs (*int*): number of epochs to train for
        optimizer (*`needle.optim.Optimizer` type*): optimizer type to use
        lr (*float*): learning rate 
        weight_decay (*float*): weight decay
        hidden_dim (*int*): hidden dim for `MLPResNet`
        data_dir (*int*): directory containing MNIST image/label files
    """
    np.random.seed(4)
    train_dataset = ndl.data.MNISTDataset(\
            "./data/train-images-idx3-ubyte.gz",
            "./data/train-labels-idx1-ubyte.gz")
    test_dataset = ndl.data.MNISTDataset(\
            "./data/t10k-images-idx3-ubyte.gz",
            "./data/t10k-labels-idx1-ubyte.gz")
    train_dataloader = ndl.data.DataLoader(\
             dataset=train_dataset,
             batch_size=batch_size,
             shuffle=True)
    test_dataloader = ndl.data.DataLoader(\
                dataset=test_dataset,
                batch_size=batch_size,
                shuffle=False)

    model = MLPResNet(784, hidden_dim)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    for i in range(epochs):
        train_err, train_loss = epoch(train_dataloader, model, opt)
    test_err, test_loss = epoch(test_dataloader, model)
    return (train_err, train_loss, test_err, test_loss)
        
    



if __name__ == "__main__":
    train_mnist(data_dir="../data")
