import torch
from torch.utils.tensorboard import SummaryWriter
class ExperimentTracker():
    def __init__(self, path):
        self.writer = SummaryWriter(path)
        print("track view:", path)
        self.scalars_dict = {}
        self.cnt_dict = {}

    def cal_ave_scalars_add_writer(self, iter):
        for name in self.scalars_dict:
            self.scalars_dict[name] /= self.cnt_dict[name]
        self.writer.add_scalars('ave scalars', self.scalars_dict, iter)
        # print(self.scalars_dict)
        # print(self.cnt_dict)
        train_metrics = ''
        for name in self.scalars_dict:
            if name[:5] == 'train':
                train_metrics += '_'+name[-4:] + '_%.3f' % (self.scalars_dict[name])
        test_metrics = ''
        for name in self.scalars_dict:
            if name[:4] == 'test':
                test_metrics += '_'+name[-4:] + '_%.3f' % (self.scalars_dict[name])


        self.scalars_dict = {}
        self.cnt_dict = {}
        return train_metrics, test_metrics

    def update_dict(self, name, val, cnt):
        # print(name, val, cnt)
        if name in self.scalars_dict:
            self.scalars_dict[name] += val
            self.cnt_dict[name] += cnt
        else:
            self.scalars_dict[name] = val
            self.cnt_dict[name] = cnt
