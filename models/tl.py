from imports import *


class TL_Model(Module):
    def __init__(self, model,starter=1024,activation=ReLU()):
        super().__init__()
        output = 1
        output_ac = Sigmoid()
        self.activation = activation
        self.lineardropout = Dropout()
        self.model = model(pretrained=True)
        self.linear1 = Linear(1000, starter)
        self.linear1batchnorm = BatchNorm1d(starter)
        self.linear2 = Linear(starter, starter*2)
        self.linear2batchnorm = BatchNorm1d(starter*2)
        self.linear3 = Linear(starter*2,starter*4)
        self.linear3batchnorm = BatchNorm1d(starter*4)
        self.linear4 = Linear(starter*4, starter*2)
        self.linear4batchnorm = BatchNorm1d(starter*2)
        self.output = Linear(starter*2, output)
        self.output_ac = output_ac

    def forward(self, X):
        preds = self.model(X)
        preds = self.lineardropout(self.linear1batchnorm(self.linear1(preds)))
        preds = self.lineardropout(self.linear2batchnorm(self.linear2(preds)))
        preds = self.lineardropout(self.linear3batchnorm(self.linear3(preds)))
        preds = self.lineardropout(self.linear4batchnorm(self.linear4(preds)))
        preds = self.output(preds)
        preds = self.output_ac(preds)
        return preds
