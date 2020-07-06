
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
PI = np.pi

class Hough(nn.Module):

    def __init__(self, angle=90, R2_list=[4, 64, 256, 1024], num_classes=80,
                 region_num=9, vote_field_size=17):
        super(Hough, self).__init__()
        self.angle = angle
        self.R2_list = R2_list
        self.region_num = region_num
        self.num_classes = num_classes
        self.vote_field_size = vote_field_size
        self.deconv_filter_padding = int(self.vote_field_size / 2)
        self.deconv_filters = self._prepare_deconv_filters()

    def  _prepare_deconv_filters(self):
        vote_center = torch.tensor([64, 64]).cuda()
        logmap = self.calculate_logmap((128, 128), vote_center)
        logmap_onehot = torch.nn.functional.one_hot(logmap.long(), num_classes=int(logmap.max()+1)).float()
        logmap_onehot = logmap_onehot[:, :, :self.region_num]
        weights = logmap_onehot / \
                        torch.clamp(torch.sum(torch.sum(logmap_onehot, dim=0), dim=0).float(), min=1.0)

        start = 63 - int(self.vote_field_size/2) + 1
        stop  = 63 + int(self.vote_field_size/2) + 2

        deconv_filters = weights[start:stop, start:stop,:].permute(2,0,1).view(self.region_num, 1,
                                                                     self.vote_field_size, self.vote_field_size)

        W = nn.Parameter(deconv_filters)
        W.requires_grad = False

        layers = []
        deconv_kernel = nn.ConvTranspose2d(
            in_channels=self.region_num,
            out_channels=1,
            kernel_size=self.vote_field_size,
            padding=self.deconv_filter_padding,
            bias=False)

        with torch.no_grad():
            deconv_kernel.weight = W

        layers.append(deconv_kernel)

        return nn.Sequential(*layers)

    def generate_grid(self, h, w):
        x = torch.arange(0, w).float().cuda()
        y = torch.arange(0, h).float().cuda()
        grid = torch.stack([x.repeat(h), y.repeat(w, 1).t().contiguous().view(-1)], 1)
        return grid.repeat(1, 1).view(-1, 2)

    def calculate_logmap(self, im_size, center, angle=90, R2_list=[4, 64, 256, 1024]):
        points = self.generate_grid(im_size[0], im_size[1])  # [x,y]
        total_angles = 360 / angle

        # check inside which circle
        y_dif = points[:, 1].cuda() - center[0].float()
        x_dif = points[:, 0].cuda() - center[1].float()

        xdif_2 = x_dif * x_dif
        ydif_2 = y_dif * y_dif
        sum_of_squares = xdif_2 + ydif_2

        # find angle
        arc_angle = (torch.atan2(y_dif, x_dif) * 180 / PI).long()

        arc_angle[arc_angle < 0] += 360

        angle_id = (arc_angle / angle).long() + 1

        c_region = torch.ones(xdif_2.shape, dtype=torch.long).cuda() * len(R2_list)

        for i in range(len(R2_list) - 1, -1, -1):
            region = R2_list[i]
            c_region[(sum_of_squares) <= region] = i

        results = angle_id + (c_region - 1) * total_angles
        results[results < 0] = 0

        return results.view(im_size[0], im_size[1])

    def forward(self, voting_map, targets=None):
        batch_size, channels, width, height = voting_map.shape
        voting_map = voting_map.view(batch_size, self.region_num, self.num_classes, width, height)
        heatmap = torch.zeros((batch_size, self.num_classes, width, height), dtype=torch.float).cuda()
        for i in range(self.num_classes):
            heatmap[:, i, :, :] = self.deconv_filters(voting_map[:, :, i, :, :]).squeeze(dim=1)

        return heatmap

