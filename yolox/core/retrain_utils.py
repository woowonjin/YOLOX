import sys
sys.path.append("/workspace/pruning/YOLOX")
sys.path.append("/workspace/pruning/netspresso-compression-toolkit")
import torch
import torch.nn as nn
from yolox.models.losses import IOUloss
from loguru import logger

class RetrainUtils(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.xin_type = torch.FloatTensor
        self.hw = [[68, 120], [34, 60], [17, 30]] # self.hw * strides = [960, 544]
        self.strides = [8, 16, 32]
        self.in_channels = [256, 512, 1024]
        self.n_anchors = 1
        self.use_l1 = False
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")
        self.grids = [torch.zeros(1).to(self.device)] * len(self.in_channels)
        self.num_classes = 11

        self.use_l1 = False

    # def split_output(self, output):
    #     return torch.split(output, [self.hw[0][0]*self.hw[0][1], self.hw[1][0]*self.hw[1][1], self.hw[2][0]*self.hw[2][1]], dim=1)
    
    def get_outputs_for_train(self, outputs):
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []
        all_outputs = []
        for k in range(len(self.hw)): # [[batch, 8160, 16], [batch, 2040, 16], [batch, 510, 16]]
            if k == 0:
                start = 0
                end = self.hw[k][0] * self.hw[k][1]
            elif k == 1:
                start = self.hw[0][0] * self.hw[0][1]
                end = self.hw[0][0] * self.hw[0][1] + self.hw[1][0] * self.hw[1][1]
            else:
                start = self.hw[0][0] * self.hw[0][1] + self.hw[1][0] * self.hw[1][1]
                end = self.hw[0][0] * self.hw[0][1] + self.hw[1][0] * self.hw[1][1] + self.hw[2][0] * self.hw[2][1]
            output, grid = self.get_output_and_grid(outputs[:, start:end, :], k, self.strides[k], self.xin_type)
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
        grid = grid.view(1, -1, 2).to(self.device) # grid : [1, 1*68*120, 2]
        # True, False -> Error
        # False, False -> OK
        # 기존 yolox-s 는 True, False
        # print(f"output : {output.get_device()}, grid : {grid.get_device()}, device : {self.device}")
        output[..., :2] = (output[..., :2] + grid) * stride # stride = 8
        # print(f"type : {output[..., 2:4].type()}, output : {output[..., 2:4]}")
        output[..., 2:4] = torch.exp(output[..., 2:4].type(self.xin_type)) * stride
        return output, grid # output : [batch, 1*68*120, 16], grid : [1, 1*68*120, 2]

    def get_losses(self, x_shifts, y_shifts, expanded_strides, labels, outputs, origin_preds, dtype):
        bbox_preds = outputs[:, :, :4]  # [batch, 10710, 4]
        obj_preds = outputs[:, :, 4].unsqueeze(-1)  # [batch, 10710, 1]
        cls_preds = outputs[:, :, 5:]  # [batch, 10710, 11]

        n_label = (labels.sum(dim=2) > 0).sum(dim=1)
        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)
        y_shifts = torch.cat(y_shifts, 1)
        expanded_strides = torch.cat(expanded_strides, 1)
        if self.use_l1:
            origin_preds = torch.cat(origin_preds, 1)
        
        cls_targets = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        fg_masks = []

        num_fg = 0.0
        num_gts = 0.0

        for batch_idx in range(outputs.shape[0]):
            num_gt = int(n_label[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                cls_target = outputs.new_zeros((0, self.num_classes)) # new_zeros -> 안의 size만큼 0으로 채우고, dtype와 device는 target tensor를 따른다.
                reg_target = outputs.new_zeros((0, 4))
                l1_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]
                gt_classes = labels[batch_idx, :num_gt, 0]
                bboxes_preds_per_image = bbox_preds[batch_idx]

                try:
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(
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
                    ) = self.get_assignments(
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
                    )


    def get_assignments(
        self,
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
        mode="gpu",
    ):
        if mode == "cpu":
            print("------------CPU Mode for This Batch-------------")
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            gt_classes = gt_classes.cpu().float()
            expanded_strides = expanded_strides.cpu().float()
            x_shifts = x_shifts.cpu()
            y_shifts = y_shifts.cpu()
        
        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            total_num_anchors,
            num_gt
        )
        print(f"fg_mask : {fg_mask.size()}, is_in_boxes_and_center : {is_in_boxes_and_center.size()}")

    def get_in_boxes_info(
        self,
        gt_bboxes_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        total_num_anchors,
        num_gt,
    ):
        gt_bboxes_per_image = gt_bboxes_per_image.to(self.device)
        expanded_strides_per_image = expanded_strides[0].to(self.device)
        # print(f"stride : {expanded_strides_per_image.get_device()}, x_shifts[0] : {x_shifts[0].get_device()}")
        x_shifts_per_image = x_shifts[0] * expanded_strides_per_image
        y_shifts_per_image = y_shifts[0]  * expanded_strides_per_image
        
        x_centers_per_image = (
            (x_shifts_per_image + 0.5 * expanded_strides_per_image)
            .unsqueeze(0)
            .repeat(num_gt, 1)
        )
        y_centers_per_image = (
            (y_shifts_per_image + 0.5 * expanded_strides_per_image)
            .unsqueeze(0)
            .repeat(num_gt, 1)
        )

        gt_bboxes_per_image_l = (
            (gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_r = (
            (gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_t = (
            (gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_b = (
            (gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )

        # 각 grid의 center로 부터 거리인듯? -> 각 object와 각 grid의 center 간의 거리 [num_fg, num_anchors]
        # print(f"x_centers_per_image : {x_centers_per_image.size()}, gt_bboxes_per_image_l : {gt_bboxes_per_image_l.size()}")
        # print(f"x_centers : {x_centers_per_image.get_device()}, gt_bboxes : {gt_bboxes_per_image.get_device()}")
        b_l = x_centers_per_image - gt_bboxes_per_image_l 
        b_r = gt_bboxes_per_image_r - x_centers_per_image 
        b_t = y_centers_per_image - gt_bboxes_per_image_t
        b_b = gt_bboxes_per_image_b - y_centers_per_image
        bboxes_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)
        # [num_gt, 10710, 4] -> 각 object의 각 grid의 center(anchor)의 left, right, top, bottom 거리

        is_in_boxes = bboxes_deltas.min(dim=-1).values > 0.0 # [num_gt, 10710]
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0 # [10710], anchor가 모든 gt box들중 하나의 gt box안에라도 들어있는 애들 -> True else False

        center_radius = 2.5

        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors # [num_gt, 10710]
        ) - center_radius * expanded_strides_per_image.unsqueeze(0) # [1, 10710]
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)


        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        is_in_centers_all = is_in_centers.sum(dim=0) > 0 # [10710], anchor가 모든 gt center box들중 하나의 gt center box안에라도 들어있는 애들 -> True else False

        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all
        is_in_boxes_and_center = (
            is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
        )
        return is_in_boxes_anchor, is_in_boxes_and_center
    
    def forward(self, preds, labels):
        preds = preds
        # splited = self.split_output(preds)
        x_shifts, y_shifts, expanded_strides, outputs, origin_preds, dtype = self.get_outputs_for_train(preds)
        self.get_losses(x_shifts, y_shifts, expanded_strides, labels, outputs, origin_preds, dtype)

        

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.load("/workspace/pruning/YOLOX/compressed_models/tiny_compressed.pt").to(device)
    dummy_input = torch.randn(16, 3, 544, 960).to(device)
    preds = model(dummy_input)
    # preds.requires_grad = True
    labels = torch.randn(16, 120, 5)
    label = torch.randint(0, 11, (16, 120)).float()
    labels[:, :, 0] = label
    labels[:, 30:, :] = 0
    criterion = RetrainUtils()
    loss = criterion(preds, labels)
    # print(preds)
    # splited = util.split_output(preds)
    # x_shifts, y_shifts, expanded_strides, outputs, origin_preds, dtype = util.get_outputs_for_train(splited)
    # print(labels.size())
    # print(f"x_shifts : {x_shifts[0].size()}, y_shifts: {y_shifts[0].size()}, expanded_strides: {expanded_strides[0].size()}, outputs: {outputs.size()}, origin_preds: {origin_preds[0].size()}, dtype: {dtype}")
    # util.get_losses(x_shifts, y_shifts, expanded_strides, labels, outputs, origin_preds, dtype)

        
