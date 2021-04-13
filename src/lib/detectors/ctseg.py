from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch
from pycocotools import  mask as mask_utils
try:
    from external.nms import soft_nms
except:
    print('NMS not imported! If you need it,'
          ' do \n cd $CenterNet_ROOT/src/lib/external \n make')
from models.decode import ctseg_decode
from models.utils import flip_tensor
from utils.image import get_affine_transform
from utils.post_process import ctseg_post_process
from utils.debugger import Debugger
from src.lib.models.losses import SegLoss

from .base_detector import BaseDetector


class CtsegDetector(BaseDetector):
    def __init__(self, opt):
        super(CtsegDetector, self).__init__(opt)
        self.seg_model = SegLoss(opt.seg_feat_channel)

    def process(self, images, return_time=False):
        with torch.no_grad():
            output = self.model(images)[-1]
            hm = output['hm'].sigmoid_()
            wh = output['wh']
            shape_feat = output['shape']
            saliency = output['saliency']
            reg = output['reg'] if self.opt.reg_offset else None
            if self.opt.flip_test:
                hm = (hm[0:1] + flip_tensor(hm[1:2])) / 2
                wh = (wh[0:1] + flip_tensor(wh[1:2])) / 2
                reg = reg[0:1] if reg is not None else None
                saliency = (saliency[0:1] + flip_tensor(saliency[1:2])) / 2
                shape_feat = shape_feat[0:1] if shape_feat is not None else None
            # assert not self.opt.flip_test,"not support flip_test"
            torch.cuda.synchronize()
            forward_time = time.time()
            dets, masks = ctseg_decode(hm, wh, shape_feat, saliency, self.seg_model, reg=reg, cat_spec_wh=self.opt.cat_spec_wh,
                                       K=self.opt.K)
        if return_time:
            return output, (dets,masks), forward_time
        else:
            return output, (dets,masks)

    def post_process(self, det_seg, meta, scale=1):
        assert scale == 1, "not support scale != 1"
        dets,seg = det_seg
        dets = dets.detach().cpu().numpy()
        seg = seg.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])
        dets = ctseg_post_process(
            dets.copy(),seg.copy(), [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'],*meta['img_size'], self.opt.num_classes)
        return dets[0]

    def merge_outputs(self, detections):
        return detections[0]


    def show_results(self, debugger, image, results):
        debugger.add_img(image, img_id='ctseg')
        for j in range(1, self.num_classes + 1):
            for i in range(len(results[j]['boxs'])):
                bbox=results[j]['boxs'][i]
                mask = mask_utils.decode(results[j]['pred_mask'][i])
                if bbox[4] > self.opt.vis_thresh:
                    debugger.add_coco_bbox(bbox[:4], j - 1, bbox[4], img_id='ctseg')
                    debugger.add_coco_seg(mask,img_id='ctseg')
        debugger.show_all_imgs(pause=self.pause)
