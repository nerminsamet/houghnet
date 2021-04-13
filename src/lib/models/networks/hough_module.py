
import torch
import torch.nn as nn

import numpy as np
PI = np.pi

class Hough(nn.Module):

    def __init__(self, angle=90, R2_list=[4, 64, 256, 1024],
                 num_classes=80, region_num=9, vote_field_size=17,
                 voting_map_size_w=128, voting_map_size_h=128, model_v1=False):
        super(Hough, self).__init__()
        self.angle = angle
        self.R2_list = R2_list
        self.region_num = region_num
        self.num_classes = num_classes
        self.vote_field_size = vote_field_size
        self.deconv_filter_padding = int(self.vote_field_size / 2)
        self.voting_map_size_w = voting_map_size_w
        self.voting_map_size_h = voting_map_size_h
        self.model_v1 = model_v1
        self.deconv_filters = self._prepare_deconv_filters()


    def  _prepare_deconv_filters(self):

        half_w = int(self.voting_map_size_w / 2)
        half_h = int(self.voting_map_size_h / 2)

        vote_center = torch.tensor([half_h, half_w]).cuda()

        logmap_onehot = self.calculate_logmap((self.voting_map_size_h, self.voting_map_size_w), vote_center)

        weights = logmap_onehot / \
                        torch.clamp(torch.sum(torch.sum(logmap_onehot, dim=0), dim=0).float(), min=1.0)

        start_x = half_h - int(self.vote_field_size/2)
        stop_x  = half_h + int(self.vote_field_size/2) + 1

        start_y = half_w - int(self.vote_field_size/2)
        stop_y  = half_w + int(self.vote_field_size/2) + 1

        '''This if-block only applies for my two pretrained models. Please ignore this for your own trainings.'''
        if self.model_v1 and self.region_num==17 and self.vote_field_size==65:
            start_x -=1
            stop_x -=1
            start_y -=1
            stop_y -=1

        deconv_filters = weights[start_x:stop_x, start_y:stop_y,:].permute(2,0,1).view(self.region_num, 1,
                                                                     self.vote_field_size, self.vote_field_size)

        W = nn.Parameter(deconv_filters.repeat(self.num_classes, 1, 1, 1))
        W.requires_grad = False

        layers = []
        deconv_kernel = nn.ConvTranspose2d(
            in_channels=self.region_num*self.num_classes,
            out_channels=1*self.num_classes,
            kernel_size=self.vote_field_size,
            padding=self.deconv_filter_padding,
            groups=self.num_classes,
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

        results.view(im_size[0], im_size[1])

        logmap = results.view(im_size[0], im_size[1])
        logmap_onehot = torch.nn.functional.one_hot(logmap.long(), num_classes=17).float()
        logmap_onehot = logmap_onehot[:, :, :self.region_num]

        return logmap_onehot

    def forward(self, voting_map, targets=None):

        if self.model_v1:
            batch_size, channels, width, height = voting_map.shape
            voting_map = voting_map.view(batch_size, self.region_num, self.num_classes, width, height)
            voting_map = voting_map.permute(0, 2, 1, 3, 4)
            voting_map = voting_map.reshape(batch_size, -1, width, height)

        heatmap = self.deconv_filters(voting_map)

        return heatmap

