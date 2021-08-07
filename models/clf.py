from imports import *


class Clf(Module):
    def __init__(self):
        super().__init__()
        self.activation = ReLU()
        self.lineardropout = Dropout()
        self.linear1 = Linear(84 * 84 * 3, 256)
        self.linear1batchnorm = BatchNorm1d(256)
        self.linear2 = Linear(256, 512)
        self.linear2batchnorm = BatchNorm1d(512)
        self.linear3 = Linear(512, 1024)
        self.linear3batchnorm = BatchNorm1d(1024)
        self.linear4 = Linear(1024, 2048)
        self.linear4batchnorm = BatchNorm1d(2048)
        self.linear5 = Linear(2048, 2048*2)
        self.linear5batchnorm = BatchNorm1d(2048*2)
        self.linear6 = Linear(2048*2, 2048)
        self.linear6batchnorm = BatchNorm1d(2048)
        self.output = Linear(2048, 1)
        self.output_ac = Sigmoid()

    def forward(self, X):
        X = X.view(-1, 84 * 84 * 3)
        preds = self.activation(
            self.lineardropout(self.linear1batchnorm(self.linear1(X)))
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
        preds = self.activation(
            self.lineardropout(self.linear6batchnorm(self.linear6(preds)))
        )
        preds = self.output_ac(self.output(preds))
        return preds
