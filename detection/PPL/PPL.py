# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn.functional as F

from mmdet.core import bbox2result
from mmdet.models.builder import DETECTORS, build_backbone, build_head, build_neck
from mmdet.models.detectors.single_stage import SingleStageDetector
from mmdet.models.detectors.two_stage import TwoStageDetector
from mmdet.utils import get_root_logger
import torch.nn as nn

from .untils import tokenize

@DETECTORS.register_module()
class PPL_MaskRCNN(TwoStageDetector):
    """
    DenseCLIP for Mask-RCNN
    """

    def __init__(self,
                 backbone,
                 text_encoder,
                 context_decoder,
                 context_length,
                 class_names,
                 seg_loss=False,
                 clip_head=True,
                 text_adapter=None,
                 tau=0.07,
                 token_embed_dim=512, 
                 text_dim=1024,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(TwoStageDetector, self).__init__(init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained

            assert text_encoder.get('pretrained') is None, \
                'both text encoder and segmentor set pretrained weight'
            text_encoder.pretrained = pretrained

        self.backbone = build_backbone(backbone)
        self.text_encoder = build_backbone(text_encoder)
        self.context_decoder = build_backbone(context_decoder)

        if neck is not None:
            self.neck = build_neck(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            roi_head.pretrained = pretrained
            self.roi_head = build_head(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.context_length = context_length
        self.tau = tau
        # add a [background] class
        self.texts = torch.cat([tokenize(c, context_length=self.context_length) for c in class_names])
        self.use_seg_loss = seg_loss
        self.class_names = class_names
        self.clip_head = clip_head

        # learnable textual contexts
        context_length = self.text_encoder.context_length - self.context_length
        self.contexts = []
        for _ in range(3):
            ctx = nn.Parameter(torch.randn(1, context_length, token_embed_dim)).cuda()
            self.contexts.append(nn.init.trunc_normal_(ctx))
        self.beta = nn.Parameter(torch.ones(80)*1.0)
        # self.contexts = nn.Parameter(torch.randn(1, context_length, token_embed_dim))
        # nn.init.trunc_normal_(self.contexts)
        self.gamma = nn.Parameter(torch.ones(text_dim) * 1e-4)

    
    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None
    def sample_gaussian_tensors(self, mu, logsigma, num_samples):
        eps = torch.randn(mu.size(0), num_samples, mu.size(1),mu.size(2), dtype=mu.dtype, device=mu.device)

        samples = eps.mul(torch.exp(logsigma.unsqueeze(1))).add_(
            mu.unsqueeze(1))
        return samples
    def _kl_divergence(self, mu,logsigma):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        losses_kl = dict()
        loss_kl = -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp()).sum() / logsigma.size(0)
        losses_kl = loss_kl*1e-3
        loss = {'loss_kl': losses_kl}
        # losses.update(add_prefix(losses_kl, 'uncertainty'))
        return loss
    def _diversity(self, text_embeddings):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        losses_div = dict()
        M = text_embeddings.size(1)
        Identity = torch.eye(M)
        text_embeddings = F.normalize(text_embeddings, dim=-1, p=2)
        text_embeddings = text_embeddings.permute(0,2,1,3)
        ScoreMat = torch.einsum('bknc, bkmc->bknm',text_embeddings,text_embeddings)
        losses_div= torch.mean(F.relu(torch.linalg.matrix_norm(ScoreMat-Identity.cuda(), dim=(-2, -1)) - self.beta[None,:]))*0.05
        loss = {'loss_diversity': losses_div}
        # losses.update(add_prefix(losses_div, 'diversity'))
        return loss
    def extract_feat(self, img, use_seg_loss=False, dummy=False):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        text_features, samples, mog_mean, mog_var = self.compute_text_features(x, dummy=dummy)
        score_maps = self.compute_score_maps(x, samples)
        x = list(x[:-1])
        x[3] = torch.cat([x[3], score_maps[3]], dim=1)
        if self.with_neck:
            x = self.neck(x)

        if use_seg_loss:
            return x, score_maps[0], text_features, mog_mean, mog_var
        else:
            return x
    
    def compute_score_maps(self, x, text_features):
        # B, K, C
        _, visual_embeddings = x[4]
        text_features = F.normalize(text_features, dim=-1)
        visual_embeddings = F.normalize(visual_embeddings, dim=1)
        score_map3 = torch.einsum('bchw,bmkc->bmkhw', visual_embeddings, text_features)
        score_map3 = torch.mean(score_map3, dim=1)/ self.tau
        score_map0 = F.upsample(score_map3, x[0].shape[2:], mode='bilinear')
        score_maps = [score_map0, None, None, score_map3]
        return score_maps

    def compute_text_features(self, x, dummy=False):
        """compute text features to each of x
        Args:
            x ([list]): list of features from the backbone, 
                x[4] is the output of attentionpool2d
        """
        global_feat, visual_embeddings = x[4]

        B, C, H, W = visual_embeddings.shape

        visual_context = torch.cat([global_feat.reshape(B, C, 1), visual_embeddings.reshape(B, C, H*W)], dim=2).permute(0, 2, 1)  # B, N, C

        # text embeddings is (B, K, C)
        if dummy:
            text_embeddings = torch.randn(B, len(self.texts), C, device=global_feat.device)
        # else:
        #     text_embeddings = self.text_encoder(self.texts.to(global_feat.device), self.contexts).expand(B, -1, -1)
        samples = []
        logsigma = []
        text_embeddings = []
        for ctx in self.contexts:
            mu_i = self.text_encoder(self.texts.to(global_feat.device), ctx).expand(B, -1, -1)
            # update text_embeddings by visual_context!
            logsigma_i = self.context_decoder(mu_i, visual_context)
            
            # samples.append(mu_i.unsqueeze(1))
            text_embeddings.append(mu_i.unsqueeze(1)) 
            logsigma.append(logsigma_i.unsqueeze(1))
        text_embeddings = torch.cat(text_embeddings, dim=1) # (B, N, C, d)
        logsigma = torch.cat(logsigma, dim=1) # (B, N, C, d)
        
        mog_mean = torch.mean(text_embeddings, dim = 1) # (B, C, d)
        mog_var = torch.log(torch.mean(text_embeddings**2 + torch.exp(logsigma)**2, dim=1) - mog_mean**2)
        samples.append(self.sample_gaussian_tensors(mu_i, logsigma_i, num_samples=5))
        samples = torch.cat(samples, dim=1)
        # text_diff = self.context_decoder(text_embeddings, visual_context)
        # text_embeddings = text_embeddings + self.gamma * text_diff

        return text_embeddings, samples, mog_mean, mog_var

    def forward_dummy(self, img):
        """Used for computing network flops.
        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img, dummy=True)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 4).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs, )
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.
            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.
        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x, score_map, text_features, mog_mean, mog_var  = self.extract_feat(img, use_seg_loss=self.use_seg_loss)
        # if self.use_seg_loss:
        #     x, score_map = x

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)
        if self.use_seg_loss:
            losses.update(self.compute_seg_loss(img, score_map, img_metas, gt_bboxes, gt_masks, gt_labels,mog_var))
            losses.update(self._kl_divergence(mog_mean, mog_var))
            losses.update(self._diversity(text_embeddings=text_features))
        return losses
    def bce_loss(self,pred, target, weight=None):
        loss=torch.sum(-(target * torch.log(pred) + (1-target) * torch.log(1-pred))*weight)
        return loss
    def compute_seg_loss(self, img, score_map, img_metas, gt_bboxes, gt_masks, gt_labels, mog_var):
        uncert = torch.exp(torch.mean(torch.mean(mog_var, dim=-1), dim=-1)) # (B)
        scale_uncert = torch.mean(torch.log(uncert)) * 0.1
        
        target, mask = self.build_seg_target(img, img_metas, gt_bboxes, gt_masks, gt_labels)
        B = mask.size(0)
        weight = mask*(1/uncert[:,None,None,None])
        loss = self.bce_loss(F.sigmoid(score_map), target, weight=weight)
        loss = loss / mask.sum()
        loss = {'loss_aux_seg': loss+scale_uncert}
        return loss

    def build_seg_target(self, img, img_metas, gt_bboxes, gt_masks, gt_labels):
        B, C, H, W = img.shape
        H //= 4
        W //= 4
        target = torch.zeros(B, len(self.class_names), H, W)
        mask = torch.zeros(B, 1, H, W)
        for i, (bboxes, gt_labels) in enumerate(zip(gt_bboxes, gt_labels)):
            bboxes = (bboxes / 4).long()
            bboxes[:, 0] = bboxes[:, 0].clamp(0, W - 1)
            bboxes[:, 1] = bboxes[:, 1].clamp(0, H - 1)
            bboxes[:, 2] = bboxes[:, 2].clamp(0, W - 1)
            bboxes[:, 3] = bboxes[:, 3].clamp(0, H - 1)
            for bbox, label in zip(bboxes, gt_labels):
                target[i, label, bbox[1]: bbox[3], bbox[0]: bbox[2]] = 1
                mask[i, :, bbox[1]: bbox[3], bbox[0]: bbox[2]] = 1
        mask = mask.expand(-1, len(self.class_names), -1, -1)
        target = target.to(img.device)
        mask = mask.to(img.device)
        return target, mask

    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = await self.rpn_head.async_simple_test_rpn(
                x, img_meta)
        else:
            proposal_list = proposals

        return await self.roi_head.async_simple_test(
            x, proposal_list, img_meta, rescale=rescale)

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""

        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.
        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        x = self.extract_feats(imgs)
        proposal_list = self.rpn_head.aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)

    def onnx_export(self, img, img_metas):

        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape
        x = self.extract_feat(img)
        proposals = self.rpn_head.onnx_export(x, img_metas)
        if hasattr(self.roi_head, 'onnx_export'):
            return self.roi_head.onnx_export(x, proposals, img_metas)
        else:
            raise NotImplementedError(
                f'{self.__class__.__name__} can not '
                f'be exported to ONNX. Please refer to the '
                f'list of supported models,'
                f'https://mmdetection.readthedocs.io/en/latest/tutorials/pytorch2onnx.html#list-of-supported-models-exportable-to-onnx'  # noqa E501
            )