from __future__ import division
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm

from torchscan import crawl_module

is_cuda = torch.cuda.is_available()

def _make_variable(X):
    variable = Variable(torch.from_numpy(X))
    if is_cuda:
        variable = variable.cuda()
    return variable


def _flatten(x):
    return x.view(x.size(0), -1)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 14 * 14, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.block1(x)
        x = _flatten(x)
        x = self.fc(x)
        return x

    def _initialize_weights(self):
        # Source: https://github.com/pytorch/vision/blob/master/torchvision/
        # models/vgg.py
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class ImageClassifier(object):

    def __init__(self):
        self.net = Net()
        if is_cuda:
            self.net = self.net.cuda()

    def _transform(self, x):
        # adding channel dimension at the first position
        x = np.expand_dims(x, axis=0)
        # bringing input between 0 and 1
        x = x / 255.
        return x

    def _get_err(self, y_pred, y_true):
        y_pred = y_pred.cpu().data.numpy().argmax(axis=1)
        y_true = y_true.cpu().data.numpy()
        return (y_pred != y_true)*100.0

    def _load_minibatch(self, img_loader, indexes):
        n_minibatch_images = len(indexes)
        X = np.zeros((n_minibatch_images, 1, 28, 28), dtype=np.float32)
        # one-hot encoding of the labels to set NN target
        y = np.zeros(n_minibatch_images, dtype=np.int)
        for i, i_load in enumerate(indexes):
            x, y[i] = img_loader.load(i_load)
            X[i] = self._transform(x)
            # since labels are [0, ..., 9], label is the same as label index
        X = _make_variable(X)
        y = _make_variable(y)
        return X, y

    def _load_test_minibatch(self, img_loader, indexes):
        n_minibatch_images = len(indexes)
        X = np.zeros((n_minibatch_images, 1, 28, 28), dtype=np.float32)
        for i, i_load in enumerate(indexes):
            x = img_loader.load(i_load)
            X[i] = self._transform(x)
        X = _make_variable(X)
        return X

    def fit(self, img_loader):
        validation_split = 0.1
        batch_size = 100
        nb_epochs = 0
        lr = 1e-4
        optimizer = optim.Adam(self.net.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss().cuda()
        if is_cuda:
            criterion = criterion.cuda()

        for epoch in range(nb_epochs):
            t0 = time.time()
            self.net.train()  # train mode
            nb_trained = 0
            nb_updates = 0
            train_loss = []
            train_err = []
            n_images = int(len(img_loader) * (1 - validation_split))
            pbar = tqdm(range(0, n_images, batch_size))
            for i in pbar:
                indexes = range(i, min(i + batch_size, n_images))
                X, y = self._load_minibatch(img_loader, indexes)
                # zero-out the gradients because they accumulate by default
                optimizer.zero_grad()
                y_pred = self.net(X)
                loss = criterion(y_pred, y)
                loss.backward()  # compute gradients
                optimizer.step()  # update params

                # Loss and accuracy
                train_err.extend(self._get_err(y_pred, y))
                train_loss.append(loss.data.item())
                nb_trained += X.size(0)
                pbar.set_description("Epoch [{}/{}], [trained {}/{}], avg_loss: {:.4f}, avg_train_err: {:.4f}".format(epoch + 1, nb_epochs, nb_trained, n_images,np.mean(train_loss), np.mean(train_err)))
                        

            self.net.eval()  # eval mode
            valid_err = []
            n_images = len(img_loader)
            while i < n_images:
                indexes = range(i, min(i + batch_size, n_images))
                X, y = self._load_minibatch(img_loader, indexes)
                i += len(indexes)
                y_pred = self.net(X)
                valid_err.extend(self._get_err(y_pred, y))
                print('here')

            delta_t = time.time() - t0
            print('Finished epoch {}'.format(epoch + 1))
            print('Time spent : {:.4f}'.format(delta_t))
            print('Train err : {:.4f}'.format(np.mean(train_err)))
            print('Valid err : {:.4f}'.format(np.mean(valid_err)))
    '''
    def predict(self, img_loader):
        # We need to batch load also at test time
        model_info = crawl_module(self.net, (1, 28, 28))
        tot_params = sum(layer['grad_params'] + layer['nograd_params'] for layer in model_info['layers'])
        tot_flops = sum(layer['flops'] for layer in model_info['layers'])
        batch_size = 32
        n_images = len(img_loader)
        i = 0
        y_proba = np.empty((n_images, 10))
        while i < n_images:
            indexes = range(i, min(i + batch_size, n_images))
            X = self._load_test_minibatch(img_loader, indexes)
            i += len(indexes)
            y_proba[indexes] = nn.Softmax()(self.net(X)).cpu().data.numpy();
        y_proba[-1,0] = tot_params#my metrics
        y_proba[-1,1] = tot_flops#my metrics
        return y_proba
    '''