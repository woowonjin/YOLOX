#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import math
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F

from yolox.utils import bboxes_iou

from .losses import IOUloss
from .network_blocks import BaseConv, DWConv


class YOLOXHead(nn.Module):
    def __init__(
        self,
        num_classes,
        width=1.0,
        strides=[8, 16, 32],
        in_channels=[256, 512, 1024],
        act="silu",
        depthwise=False,
    ):
        """
        Args:
            act (str): activation type of conv. Defalut value: "silu".
            depthwise (bool): whether apply depthwise conv in conv branch. Defalut value: False.
        """
        super().__init__()

        self.n_anchors = 1
        self.num_classes = num_classes
        self.decode_in_inference = True  # for deploy, set to False

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()
        Conv = DWConv if depthwise else BaseConv

        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(
                    in_channels=int(in_channels[i] * width),
                    out_channels=int(256 * width),
                    ksize=1,
                    stride=1,
                    act=act,
                )
            )
            self.cls_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.cls_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * self.num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.reg_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=4,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.obj_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * 1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )

        self.use_l1 = False
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")
        self.strides = strides
        self.grids = [torch.zeros(1)] * len(in_channels)

    def initialize_biases(self, prior_prob):
        for conv in self.cls_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.obj_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, xin, labels=None, imgs=None):
        outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []
        # print(f"labels : {labels}")
        for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
            zip(self.cls_convs, self.reg_convs, self.strides, xin)
        ):
            x = self.stems[k](x) # 256 -> 256, 512 -> 256, 1024 -> 256 (channel)
            cls_x = x
            reg_x = x

            cls_feat = cls_conv(cls_x) # cls_conv : 256 -> 256 (channel)
            cls_output = self.cls_preds[k](cls_feat) # cls_preds : 256 -> n_anchor * num_classes (channel)

            reg_feat = reg_conv(reg_x) # reg_conv : 256 -> 256 (channel)
            reg_output = self.reg_preds[k](reg_feat) # reg_preds : 256 -> 4 (channel)
            obj_output = self.obj_preds[k](reg_feat) # obj_preds : 256 -> n_anchor * 1 (channel)

             # outputs : [tensor([batch, 16, 68, 120]), tensor([batch, 16, 34, 60]), tensor([batch, 16, 17, 30])]
            if self.training:
                output = torch.cat([reg_output, obj_output, cls_output], 1) # [batch, 16, 68, 120]
                output, grid = self.get_output_and_grid( # strides=[8, 16, 32]
                    output, k, stride_this_level, xin[0].type()
                ) # output : [batch, 1*68*120, 16], grid : [1, 1*68*120, 2]
                x_shifts.append(grid[:, :, 0]) # grid : [... [x, y]]
                y_shifts.append(grid[:, :, 1])
                expanded_strides.append(
                    torch.zeros(1, grid.shape[1])
                    .fill_(stride_this_level)
                    .type_as(xin[0])
                ) # expanded_strides : [[8, 8, 8, 8, .....], [16, 16, 16, ....], [32, 32, 32, ....]]
                if self.use_l1:
                    batch_size = reg_output.shape[0]
                    hsize, wsize = reg_output.shape[-2:]
                    reg_output = reg_output.view(
                        batch_size, self.n_anchors, 4, hsize, wsize
                    ) # [batch, 1, 4, 68, 120] -> 여기서 4는 x, y, width, height인듯
                    reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(
                        batch_size, -1, 4
                    ) # [batch, 1, 4, 68, 120] -> [batch, 1, 68, 120, 4] -> [batch, 1*68*120, 4] 
                    origin_preds.append(reg_output.clone())

            else:
                # print(f"reg : {reg_output.size()}, obj : {obj_output.size()} cls : {cls_output.size()}")
                output = torch.cat(
                    [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1
                )
            # 다른점 : training시에는 get_output_and_grid함수에서 output의 (x, y, w, h) 에 grid값을 더해주고 stride값을 곱해주지만
            # eval모드에서는 그렇지 않고 obj_output과 cls_output에 sigmoid를 해준다.

            outputs.append(output)
        # training_mode outputs : [tensor([batch, 1*68*120, 16]), tensor([batch, 1*34*60, 16]), tensor([batch, 1*17*30, 16])]
        # eval_mode outputs : [tensor([batch, 16, 68, 120]), tensor([batch, 16, 34, 60]), tensor([batch, 16, 17, 30])]
        if self.training:
            return self.get_losses(
                imgs, # [batch, 3, 544, 960]
                x_shifts, # [[1, 1*68*120], [1, 1*34*60], [1, 1*17*30]]
                y_shifts, # [[1, 1*68*120], [1, 1*34*60], [1, 1*17*30]]
                expanded_strides, # expanded_strides : [[8]*8160, [16]*2040, [32]*510]
                labels, # [batch, 120, 5]
                torch.cat(outputs, 1), # [batch, 10710, 16]
                origin_preds, # [[batch, 1*68*120, 4], [batch, 1*34*60, 4], [batch, 1*17*30, 4]]
                dtype=xin[0].dtype,
            )
        # elif self.compress_mode:
            # outputs = torch.cat(
                # [x.flatten(start_dim=2) for x in outputs], dim=2
            # ).permute(0, 2, 1) # [tensor([batch, 16, 68, 120]), tensor([batch, 16, 34, 60]), tensor([batch, 16, 17, 30])]
                                #   -> [ [batch, 16, 8160], [batch, 16, 2040] , [batch, 16, 510]] -> [batch, 16, 10710] -> [batch, 10710, 16]
            # print("="*50)
            # print("self.compress_mode == True")
            # print("="*50)
            # return outputs 
        else:
            self.hw = [x.shape[-2:] for x in outputs] # [[68, 120], [34, 60], [17, 30]]
            # [batch, n_anchors_all, 85]
            outputs = torch.cat(
                [x.flatten(start_dim=2) for x in outputs], dim=2
            ).permute(0, 2, 1)
            if self.decode_in_inference:
                print("="*50)
                print("self.compress_mode == False")
                print("self.decode_in_inference == True")
                print("="*50)
                return self.decode_outputs(outputs, dtype=xin[0].type())
            else:
                print("="*50)
                print("self.compress_mode == False")
                print("self.decode_in_inference == False")
                print("="*50)
                return outputs

    def get_output_and_grid(self, output, k, stride, dtype):

        grid = self.grids[k] # tensor([0.])

        # output : [batch, 16, 68, 120]
        batch_size = output.shape[0]
        n_ch = 5 + self.num_classes # [x, y, width, height, confidence, classes] ?
        hsize, wsize = output.shape[-2:] # [68, 120]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)]) 
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype) # [ [ [ [[0, 0], [1, 0], [2,0] ...], [ [0, 1], [1, 1]... ] ] ] ]
            self.grids[k] = grid # [1, 1, 68, 120, 2]
        # n_anchors = 1 
        output = output.view(batch_size, self.n_anchors, n_ch, hsize, wsize) # output : [batch, 1, 16, 68, 120]
        output = output.permute(0, 1, 3, 4, 2).reshape(
            batch_size, self.n_anchors * hsize * wsize, -1
        ) # output : [batch, 1, 16, 68, 120] -> [batch, 1, 68, 120, 16] -> [batch, 1*68*120, 16]
        grid = grid.view(1, -1, 2) # grid : [1, 1*68*120, 2]
        output[..., :2] = (output[..., :2] + grid) * stride # stride = 8
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
        return output, grid # output : [batch, 1*68*120, 16], grid : [1, 1*68*120, 2]

    def decode_outputs(self, outputs, dtype):
        #outputs : [batch, 10710, 16]
        grids = []
        strides = []
        # self.hw : [[68, 120], [34, 60], [17, 30]], self.strides : [8, 16, 32]
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride)) # torch.full(size->tuple or list, fill_value) -> value로 size tensor만큼을 채운다.

        grids = torch.cat(grids, dim=1).type(dtype) # [1, 10710, 2]
        strides = torch.cat(strides, dim=1).type(dtype) # [[8, 8, 8, 8, .....], [16, 16, 16, ....], [32, 32, 32, ....]]

        outputs[..., :2] = (outputs[..., :2] + grids) * strides
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
        return outputs

    def get_losses(
        self,
        imgs, # [batch, 3, 544, 960]
        x_shifts, # [[1, 1*68*120, 1], [1, 1*34*60, 1], [1, 1*17*30, 1]]
        y_shifts, # [[1, 1*68*120, 1], [1, 1*34*60, 1], [1, 1*17*30, 1]]
        expanded_strides, # expanded_strides : [[8]*8160, [16]*2040, [32]*510]
        labels, # [batch, 120, 5]
        outputs, # [batch, 10710, 16]
        origin_preds, # [[batch, 1*68*120, 4], [batch, 1*34*60, 4], [batch, 1*17*30, 4]]
        dtype,
    ):
        bbox_preds = outputs[:, :, :4]  # [batch, 10710, 4]
        obj_preds = outputs[:, :, 4].unsqueeze(-1)  # [batch, 10710, 1]
        cls_preds = outputs[:, :, 5:]  # [batch, 10710, 11]

        # calculate targets
        # labels : [batch, 120, 5] -> 5 : label, xmin, ymin, xmax, ymax
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects -> object이 없으면 [label, xmin, ymin, xmax, ymax] == [0, 0, 0, 0, 0]

        total_num_anchors = outputs.shape[1] # 10710
        x_shifts = torch.cat(x_shifts, 1)  # [1, 10710]
        y_shifts = torch.cat(y_shifts, 1)  # [1, 10710]
        expanded_strides = torch.cat(expanded_strides, 1) # [1, 10710] -> [[8, 8, 8, ..., 16, 16, 16, .., 32, 32, 32]]
        if self.use_l1:
            origin_preds = torch.cat(origin_preds, 1)

        cls_targets = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        fg_masks = []

        num_fg = 0.0
        num_gts = 0.0

        for batch_idx in range(outputs.shape[0]): # outputs : [batch, 10710, 16]
            num_gt = int(nlabel[batch_idx]) # number of ground truth
            num_gts += num_gt
            if num_gt == 0:
                cls_target = outputs.new_zeros((0, self.num_classes)) # new_zeros -> 안의 size만큼 0으로 채우고, dtype와 device는 target tensor를 따른다.
                reg_target = outputs.new_zeros((0, 4))
                l1_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                # labels : [2, 120, 5]
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5] # [num_gt, 4]
                gt_classes = labels[batch_idx, :num_gt, 0] # [num_gt]
                bboxes_preds_per_image = bbox_preds[batch_idx] # bbox_preds -> [batch, 10710, 4] -> [10710, 4]

                try:
                    (
                        gt_matched_classes, # preds와 매칭된 classes들 -> [num_fg]
                        fg_mask, # grid의 중심이 gt_bbox안에 들어오거나, cneter_radius안에 들어오는 anchor index [True or False] -> [10710] foreground
                        pred_ious_this_matching, # preds중에서 gt와 매칭된 애들과 그 애들의 ious -> [num_fg]
                        matched_gt_inds, # 매칭된 gt들의 indexes -> [num_fg]
                        num_fg_img, # pred들 중 gt와 매칭이 된 애들의 갯수
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        total_num_anchors, # 10710
                        gt_bboxes_per_image, # [num_gt, 4]
                        gt_classes, # [num_gt, 1]
                        bboxes_preds_per_image, # [10710, 4]
                        expanded_strides, # [1, 10710] -> [[8, 8, 8, ..., 16, 16, 16, .., 32, 32, 32]]
                        x_shifts, # [1, 10710]
                        y_shifts, # [1, 10710]
                        cls_preds, # [batch, 10710, 11]
                        bbox_preds, # [batch, 10710, 4]
                        obj_preds, # [batch, 10710, 1]
                        labels, # [batch, 120, 5]
                        imgs, # [batch, 3, 544, 960]
                    )
                except RuntimeError:
                    logger.error(
                        "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                           CPU mode is applied in this batch. If you want to avoid this issue, \
                           try to reduce the batch size or image size."
                    )
                    torch.cuda.empty_cache()
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds, 
                        obj_preds,
                        labels,
                        imgs,
                        "cpu",
                    )

                torch.cuda.empty_cache()
                num_fg += num_fg_img

                cls_target = F.one_hot(
                    gt_matched_classes.to(torch.int64), self.num_classes
                ) * pred_ious_this_matching.unsqueeze(-1) # [num_fg_img, 11] * [num_fg_img, 1] = [num_fg_img, 11]
                obj_target = fg_mask.unsqueeze(-1) # [10710, 1]
                reg_target = gt_bboxes_per_image[matched_gt_inds] # [num_fg_img, 4]
                if self.use_l1:
                    l1_target = self.get_l1_target(
                        outputs.new_zeros((num_fg_img, 4)),
                        gt_bboxes_per_image[matched_gt_inds],
                        expanded_strides[0][fg_mask],
                        x_shifts=x_shifts[0][fg_mask],
                        y_shifts=y_shifts[0][fg_mask],
                    )

            cls_targets.append(cls_target) # [num_fg_img, 11]
            reg_targets.append(reg_target) # [num_fg_img, 4]
            obj_targets.append(obj_target.to(dtype)) # [10710, 1]
            fg_masks.append(fg_mask) # [10710]
            print(f"cls_target : {cls_target.size()}, reg_target : {reg_target.size()}, obj_target : {obj_target.size()}, fg_mask : {fg_mask.size()}, dtype : {dtype}")
            if self.use_l1:
                l1_targets.append(l1_target)

        cls_targets = torch.cat(cls_targets, 0) # [num_fg, 11]
        reg_targets = torch.cat(reg_targets, 0) # [num_fg, 4]
        obj_targets = torch.cat(obj_targets, 0) # [10710*batch, 1]
        fg_masks = torch.cat(fg_masks, 0) # [10710*batch]
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)

        num_fg = max(num_fg, 1)
        loss_iou = (
            self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)
        ).sum() / num_fg
        loss_obj = (
            self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)
        ).sum() / num_fg
        loss_cls = (
            self.bcewithlog_loss(
                cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets
            )
        ).sum() / num_fg
        if self.use_l1:
            loss_l1 = (
                self.l1_loss(origin_preds.view(-1, 4)[fg_masks], l1_targets)
            ).sum() / num_fg
        else:
            loss_l1 = 0.0

        reg_weight = 5.0
        loss = reg_weight * loss_iou + loss_obj + loss_cls + loss_l1

        return (
            loss,
            reg_weight * loss_iou,
            loss_obj,
            loss_cls,
            loss_l1,
            num_fg / max(num_gts, 1),
        )

    def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-8):
        l1_target[:, 0] = gt[:, 0] / stride - x_shifts
        l1_target[:, 1] = gt[:, 1] / stride - y_shifts
        l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)
        l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)
        return l1_target

    @torch.no_grad()
    def get_assignments(
        self,
        batch_idx,
        num_gt,
        total_num_anchors, # 10710
        gt_bboxes_per_image, # [num_gt, 4]
        gt_classes, # [num_gt]
        bboxes_preds_per_image, # [10710, 4]
        expanded_strides, # [1, 10710] -> [[8, 8, 8, ..., 16, 16, 16, .., 32, 32, 32]]
        x_shifts, # [1, 10710]
        y_shifts, # [1, 10710]
        cls_preds, # [batch, 10710, 11]
        bbox_preds, # [batch, 10710, 4]
        obj_preds, # [batch, 10710, 1]
        labels, # [batch, 120, 5]
        imgs, # [batch, 3, 544, 960]
        mode="gpu",
    ):
        # print(f"obj_preds : {obj_preds.size()}")
        if mode == "cpu":
            print("------------CPU Mode for This Batch-------------")
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            gt_classes = gt_classes.cpu().float()
            expanded_strides = expanded_strides.cpu().float()
            x_shifts = x_shifts.cpu()
            y_shifts = y_shifts.cpu()

        # fg_mask : [10710], grid의 중심이 gt_bbox안에 들어오거나, cneter_radius안에 들어오는 anchor index [True or False]
        # fg_mask에서 True의 갯수 -> fg_num
        # is_in_boxes_and_center : [num_gt, fg_num] grid의 중심이 gt_bbox안에 들어오고 and cneter_radius안에 들어오는 anchor index [True or False]
        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
            gt_bboxes_per_image, # [num_gt, 4]
            expanded_strides, # [1, 10710] -> [[8, 8, 8, ..., 16, 16, 16, .., 32, 32, 32]]
            x_shifts, # [1, 10710]
            y_shifts, # [1, 10710]
            total_num_anchors, # 10710
            num_gt,
        )
        # print(f"is_in_boxes_and_center : {is_in_boxes_and_center.size()}, fg_mask : {fg_mask.size()}")
        
        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask] # [fg_num, 4]
        cls_preds_ = cls_preds[batch_idx][fg_mask] # [fg_num, 11]
        obj_preds_ = obj_preds[batch_idx][fg_mask] # [fg_num, 1]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0] # fg_num

        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()

        # gt_bboxes_per_image : [num_gt, 4], bboxes_preds_per_image : [fg_num, 4]
        pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False) # IOU 값 -> [num_gt, fg_num]
        # print(f"prev_size : {F.one_hot(gt_classes.to(torch.int64), self.num_classes).size()}")
        # print(f"num_in_boxes_anchor : {num_in_boxes_anchor}")
        # print(f"gt_classes : {gt_classes.size()}")
        gt_cls_per_image = (
            F.one_hot(gt_classes.to(torch.int64), self.num_classes)
            .float()
            .unsqueeze(1)
            .repeat(1, num_in_boxes_anchor, 1)
        ) # [num_gt, num_inboxes_anchor, 11]
        # print(f"gt_size : {gt_cls_per_image.size()}, num_gt : {gt_classes.size()}, num_inboxes_anchor : {num_in_boxes_anchor}")
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8) # [num_gt, fg_num]

        if mode == "cpu":
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()

        with torch.cuda.amp.autocast(enabled=False):
            # cls_preds : [fg_num, 11]
            # obj_preds : [fg_num, 1]
            cls_preds_ = (
                cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_() # [num_gt, fg_num, 11]
                * obj_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_() # [num_gt, fg_num, 1]
            )
            pair_wise_cls_loss = F.binary_cross_entropy(
                cls_preds_.sqrt_(), gt_cls_per_image, reduction="none"
            ).sum(-1) # [num_gt, fg_num]
            # print(f"pair_wise_cls_loss : {pair_wise_cls_loss.size()}")
        del cls_preds_

        cost = (
            pair_wise_cls_loss # [num_gt, fg_num]
            + 3.0 * pair_wise_ious_loss # [num_gt, fg_num]
            + 100000.0 * (~is_in_boxes_and_center) # [num_gt, fg_num] -> False일때 cost 계산
        )

        (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
        ) = self.dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask) # gt_classes : [num_gt], fg_mask : [10710], pair_wise_ious : [num_gt, fg_num], cost : [num_gt, fg_num]
        # num_fg : pred들 중 gt와 매칭이 된 애들의 갯수
        # gt_matched_classes : preds와 매칭된 classes들 [num_fg]
        # pred_ious_this_matching : preds중에서 gt와 매칭된 애들과 그 애들의 ious # [num_fg]
        # matched_gt_inds : 매칭된 gt들의 indexes -> [num_fg]

        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        if mode == "cpu":
            gt_matched_classes = gt_matched_classes.cuda()
            fg_mask = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()

        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )

    def get_in_boxes_info(
        self,
        gt_bboxes_per_image, # [num_gt, 4]
        expanded_strides, # [1, 10710] -> [[8, 8, 8, ..., 16, 16, 16, .., 32, 32, 32]]
        x_shifts, # [1, 10710]
        y_shifts, # [1, 10710]
        total_num_anchors, # 10710
        num_gt,
    ):
        expanded_strides_per_image = expanded_strides[0] # [10710]
        x_shifts_per_image = x_shifts[0] * expanded_strides_per_image # [10710]
        y_shifts_per_image = y_shifts[0] * expanded_strides_per_image # [10710]
        x_centers_per_image = (
            (x_shifts_per_image + 0.5 * expanded_strides_per_image)
            .unsqueeze(0)
            .repeat(num_gt, 1)
        )  # [n_anchor] -> [n_gt, n_anchor] -> [n_gt, 10710]
        # print(f"gt_bboxes_per_image : {gt_bboxes_per_image.size()}")
        # print(f"gt_bboxes_per_image : {gt_bboxes_per_image}")
        y_centers_per_image = (
            (y_shifts_per_image + 0.5 * expanded_strides_per_image)
            .unsqueeze(0)
            .repeat(num_gt, 1)
        ) # [n_anchor] -> [n_gt, n_anchor] -> [n_gt, 10710]

        # gt_bboxes_per_image : [x_center, y_center, w, h]
        gt_bboxes_per_image_l = (
            (gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        ) # [num_gt, 10710]
        gt_bboxes_per_image_r = (
            (gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        ) # [num_gt, 10710]
        gt_bboxes_per_image_t = (
            (gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        ) # [num_gt, 10710]
        gt_bboxes_per_image_b = (
            (gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        ) # [num_gt, 10710]

        # 전체과정 : grid들 중 grid의 중심이 ground_truth box안에 들어오는 grid들만 고른다.
        b_l = x_centers_per_image - gt_bboxes_per_image_l # [num_gt, 10710] -> 중심에서의 거리?
        b_r = gt_bboxes_per_image_r - x_centers_per_image # [num_gt, 10710]
        b_t = y_centers_per_image - gt_bboxes_per_image_t # [num_gt, 10710]
        b_b = gt_bboxes_per_image_b - y_centers_per_image # [num_gt, 10710]
        bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2) # [num_gt, 10710, 4]

        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0 # [num_gt, 10710] -> b_l, b_t, b_r, b_b는 모두 양수여야 한다. 그래서 모두 양수인걸 찾고
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0 # [10710] -> 모든 gt에 대해 1개라도 만족하면(양수) 후보가 된다.
        # in fixed center

        center_radius = 2.5

        # gt_bboxes_per_image : [num_gt, 4] -> [center_x, center_y, w, h]
        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors # [num_gt, 10710]
        ) - center_radius * expanded_strides_per_image.unsqueeze(0) # [num_gt, 10710]
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0) # [num_gt, 10710]
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0) # [num_gt, 10710]
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0) # [num_gt, 10710]
        
        # grid의 중심중 ground_truth의 center_radius안에 들어오는 grid만 고른다.
        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0 # [num_gt, 10710] -> c_l, c_t, c_r, c_b는 모두 양수여야 한다. 그래서 모두 양수인걸 찾고
        is_in_centers_all = is_in_centers.sum(dim=0) > 0 # [10710] -> 모든 gt에 대해 1개라도 만족하면(양수) 후보가 된다.

        # in boxes and in centers
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all # [10710] # grid의 중심이 gt_bbox안에 들어오거나, cneter_radius안에 들어오는 anchor
        # print(f"is_in_boxes_anchor : {is_in_boxes_anchor.size()}")
        is_in_boxes_and_center = (
            is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
        ) # grid의 중심이 gt_bbox안에 들어오고 and cneter_radius안에 들어오는 anchor -> True, 둘중 하나라도 만족 못하면 False -> [num_gt, 만족하는 갯수]
        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        # Dynamic K
        # ---------------------------------------------------------------
        # cost : [num_gt, fg_num], pair_wise_ious : [num_gt, fg_num], gt_classes : [num_gt], fg_mask : [10710],
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8) # [num_gt, fg_num]
        # pair_wise_ious -> 각 gt와 각 preds에 대한 iou loss -> iou가 0에 가까울수록 크고, iou가 1에 가까울수록 작다.
        ious_in_boxes_matrix = pair_wise_ious # [num_gt, fg_num]
        n_candidate_k = min(10, ious_in_boxes_matrix.size(1))
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1) # [num_gt, n_candidate_k] -> iou가 0에 가까운 애들 뽑기
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1) # [num_gt]
        dynamic_ks = dynamic_ks.tolist() # len == num_gt 짜리 int list -> 제일 작은값은 1보다 크거나 같다.
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx], largest=False # cost[gt_idx]에서 k개의 가장 cost가 작은 elements의 idx 리턴
            )
            matching_matrix[gt_idx][pos_idx] = 1


        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0) # [fg_num] -> 예측에 맞는 gt가 몇개 있는지
        if (anchor_matching_gt > 1).sum() > 0:
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0) # 각 pred와의 cost가 가장 작은 gt의 index 반환
            matching_matrix[:, anchor_matching_gt > 1] *= 0 # 예측에 맞는 gt가 1개 보다 많은 것들에 대해서 matching_matrix를 0으로 초기화
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1 # 각 pred_idx와 cost가 가장 작은 gt_idx의 matching_matrix[gt_idx, pred_idx] = 1
        fg_mask_inboxes = matching_matrix.sum(0) > 0 # pred중에서 gt와 matching이 되는 애들이 있는 pred들
        num_fg = fg_mask_inboxes.sum().item() # pred중에서 gt와 matching이 되는 애들이 있는 pred들의 갯수

        # fg_mask : [10710], grid의 중심이 gt_bbox안에 들어오거나, cneter_radius안에 들어오는 anchor index [True or False]
        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0) # pred각각에 gt와의 매칭중 가장 매칭이 잘된 애들의 index를 리턴
        gt_matched_classes = gt_classes[matched_gt_inds]
        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]
        # print(f"num_fg : {num_fg}, gt_matched_classes : {gt_matched_classes.size()}, pred_ious_this_matching : {pred_ious_this_matching.size()}, matched_gt_inds : {matched_gt_inds.size()}")
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds
        # num_fg : pred들 중 gt와 매칭이 된 애들의 갯수
        # gt_matched_classes : preds와 매칭된 classes들 [num_fg]
        # pred_ious_this_matching : preds중에서 gt와 매칭된 애들과 그 애들의 ious # [num_fg]
        # matched_gt_inds : 매칭된 gt들의 indexes -> [num_fg]
