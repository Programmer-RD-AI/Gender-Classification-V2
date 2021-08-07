from imports import *


class CNN(Module):
    def __init__(self):
        super().__init__()
        self.activation = ReLU()
        self.max_pool2d = MaxPool2d((2, 2), (2, 2))
        self.convdropout = Dropout()
        self.conv1 = Conv2d(3, 8, (5, 5), (1, 1))
        self.conv1batchnorm = BatchNorm2d(8)
        self.conv2 = Conv2d(8, 16, (5, 5), (1, 1))
        self.conv2batchnorm = BatchNorm2d(16)
        self.conv3 = Conv2d(16, 32, (5, 5), (1, 1))
        self.conv3batchnorm = BatchNorm2d(32)
        self.lineardropout = Dropout()
        self.linear1 = Linear(32 * 7 * 7, 128)
        self.linear1batchnorm = BatchNorm1d(128)
        self.linear2 = Linear(128, 256)
        self.linear2batchnorm = BatchNorm1d(256)
        self.linear3 = Linear(256, 512)
        self.linear3batchnorm = BatchNorm1d(512)
        self.linear4 = Linear(512, 1024)
        self.linear4batchnorm = BatchNorm1d(1024)
        self.linear5 = Linear(1024, 512)
        self.linear5batchnorm = BatchNorm1d(512)
        self.output = Linear(512, 1)
        self.output_ac = Sigmoid()

    def forward(self, X):
        preds = self.activation(
            self.convdropout(self.max_pool2d(self.conv1batchnorm(self.conv1(X))))
        )
        preds = self.activation(
            self.convdropout(self.max_pool2d(self.conv2batchnorm(self.conv2(preds))))
        )
        preds = self.activation(
            self.convdropout(self.max_pool2d(self.conv3batchnorm(self.conv3(preds))))
        )
        preds = preds.view(-1, 32 * 7 * 7)
        preds = self.activation(
            self.lineardropout(self.linear1batchnorm(self.linear1(preds)))
        )
        preds = self.activation(
            self.lineardropout(self.linear2batchnorm(self.linear2(preds)))
        )
        preds = self.activation(
            self.lineardropout(self.linear3batchnorm(self.linear3(preds)))
        )
        preds = self.activation(
            self.lineardropout(self.linear4batchnorm(self.linear4(preds)))
        )
        preds = self.activation(
            self.lineardropout(self.linear5batchnorm(self.linear5(preds)))
        )
        preds = self.output_ac(self.output(preds))
        return preds
