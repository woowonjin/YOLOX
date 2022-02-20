import sys
sys.path.append("/workspace/pruning/YOLOX")

import torch
import torch.nn as nn
from yolox.models.losses import IOUloss

class RetrainUtils:
    def __init__(self):
        xin_type = torch.FloatTensor
        self.hw = [[68, 120], [34, 60], [17, 30]]
        self.strides = [8, 16, 32]
        self.in_channels=[256, 512, 1024],

        self.use_l1 = False
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")
        self.grids = [torch.zeros(1)] * len(self.in_channels)

    def split_output(self, output):
        return torch.split(output, [self.hw[0][0]*self.hw[0][1], self.hw[1][0]*self.hw[1][1], self.hw[2][0]*self.hw[2][1]], dim=1)
    
    
        

if __name__ == "__main__":
    util = RetrainUtils()
    preds = torch.randn(16, 10710, 16)
    splited = util.split_output(preds)
    # for temp in splited:
    #     print(temp.size())
