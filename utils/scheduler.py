# -*- coding:utf-8 -*-
import torch

class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)


def get_scheduler(optimizer, n_epochs, offset, decay_start_epoch):
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=LambdaLR(n_epochs, offset, decay_start_epoch).step)