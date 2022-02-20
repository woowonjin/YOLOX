import sys
sys.path.append("/workspace/pruning/YOLOX")

import torch
import torch.nn as nn
from yolox.models.losses import IOUloss

class RetrainUtils:
    def __init__(self):
        self.xin_type = torch.FloatTensor
        self.hw = [[68, 120], [34, 60], [17, 30]] # self.hw * strides = [960, 544]
        self.strides = [8, 16, 32]
        self.in_channels = [256, 512, 1024]
        self.n_anchors = 1
        self.use_l1 = False
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")
        self.grids = [torch.zeros(1)] * len(self.in_channels)
        self.num_classes = 11

        self.use_l1 = False

    def split_output(self, output):
        return torch.split(output, [self.hw[0][0]*self.hw[0][1], self.hw[1][0]*self.hw[1][1], self.hw[2][0]*self.hw[2][1]], dim=1)
    
    def get_outputs_for_train(self, outputs):
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []
        all_outputs = []
        for k, output in enumerate(outputs): # [[batch, 8160, 16], [batch, 2040, 16], [batch, 510, 16]]
            output, grid = self.get_output_and_grid(output, k, self.strides[k], self.xin_type)
            x_shifts.append(grid[:, :, 0]) # grid : [... [x, y]]
            y_shifts.append(grid[:, :, 1])
            expanded_strides.append(
                torch.zeros(1, grid.shape[1])
                .fill_(self.strides[k])
                .type(self.xin_type)
            )
            if self.use_l1:
                batch_size = output.shape[0]
                reg_output = output[:, :, :4].clone()
                origin_preds.append(reg_output)
            all_outputs.append(output)
        return x_shifts, y_shifts, expanded_strides, torch.cat(all_outputs, 1), origin_preds, self.xin_type


    def get_output_and_grid(self, output, k, stride, dtype):
        grid = self.grids[k]
        batch_size = output.shape[0]
        hsize, wsize = self.hw[k]
        # output : [batch, 68*120, 16]
        if grid.shape[2:4] != torch.Size([hsize, wsize]):
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)], indexing=None)
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype) # [ [ [ [[0, 0], [1, 0], [2,0] ...], [ [0, 1], [1, 1]... ] ] ] ]
            self.grids[k] = grid # [1, 1, 68, 120, 2]
        grid = grid.view(1, -1, 2) # grid : [1, 1*68*120, 2]
        output[..., :2] = (output[..., :2] + grid) * stride # stride = 8
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
        return output, grid # output : [batch, 1*68*120, 16], grid : [1, 1*68*120, 2]




        

if __name__ == "__main__":
    util = RetrainUtils()
    preds = torch.randn(16, 10710, 16)
    splited = util.split_output(preds)
    x_shifts, y_shifts, expanded_strides, outputs, origin_preds, dtype = util.get_outputs_for_train(splited)
    # print(f"x_shifts : {x_shifts[0].size()}, y_shifts: {y_shifts[0].size()}, expanded_strides: {expanded_strides[0].size()}, outputs: {outputs.size()}, origin_preds: {origin_preds[0].size()}, dtype: {dtype}")
    
        
