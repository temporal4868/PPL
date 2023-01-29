from operator import gt
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.core import add_prefix
from mmseg.ops import resize
from mmseg.models import builder
from mmseg.models.builder import SEGMENTORS
from mmseg.models.segmentors.base import BaseSegmentor

from .untils import tokenize


@SEGMENTORS.register_module()
class PPL(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone,
                 text_encoder,
                 vcp_decoder,
                 decode_head,
                 class_names,
                 context_length,
                 context_feature='attention',
                 score_concat_index=3,
                 text_head=False,
                 neck=None,
                 tau=0.07,
                 auxiliary_head=None,
                 identity_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 token_embed_dim=512, text_dim=1024,
                 **args):
        super(PPL, self).__init__(init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained

            assert text_encoder.get('pretrained') is None, \
                'both text encoder and segmentor set pretrained weight'
            
            if 'RN50' not in pretrained and 'RN101' not in pretrained and 'ViT-B' not in pretrained:
                print('not CLIP pre-trained weight, using CLIP ViT-B-16')
                text_encoder.pretrained = 'pretrained/ViT-B-16.pt'
            else:
                text_encoder.pretrained = pretrained

        self.backbone = builder.build_backbone(backbone)
        self.text_encoder = builder.build_backbone(text_encoder)
        self.vcp_decoder = builder.build_backbone(vcp_decoder)
        self.context_length = context_length
        self.score_concat_index = score_concat_index

        assert context_feature in ['attention', 'backbone']
        self.context_feature = context_feature

        self.text_head = text_head
        self.tau = tau

        if neck is not None:
            self.neck = builder.build_neck(neck)

        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.with_identity_head = False
        self.identity_head = None
        self._init_identity_head(identity_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.texts = torch.cat([tokenize(c, context_length=self.context_length) for c in class_names])
        self.num_classes = len(self.texts)


        context_length = self.text_encoder.context_length - self.context_length
        self.contexts = []
        for _ in range(3):
            ctx = nn.Parameter(torch.randn(1, context_length, token_embed_dim, device="cuda")).cuda()
            self.contexts.append(nn.init.trunc_normal_(ctx))
        self.contexts = nn.ParameterList(self.contexts)
        # self.contexts = torch.stack(self.contexts)
        self.beta = nn.Parameter(torch.ones(150)*1.0)
        assert self.with_decode_head
    def sample_gaussian_tensors(self, mu, logsigma, num_samples):
        eps = torch.randn(mu.size(0), num_samples, mu.size(1),mu.size(2), dtype=mu.dtype, device=mu.device)

        samples = eps.mul(torch.exp(logsigma.unsqueeze(1))).add_(
            mu.unsqueeze(1))
        return samples
    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(builder.build_head(head_cfg))
            else:
                self.auxiliary_head = builder.build_head(auxiliary_head)
    
    def _init_identity_head(self, identity_head):
        """Initialize ``auxiliary_head``"""
        if identity_head is not None:
            self.with_identity_head = True
            self.identity_head = builder.build_head(identity_head)

    def extract_feat(self, img):
        """Extract features from images."""
        x = self.backbone(img)
        return x

    def _decode_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.forward_train(x, img_metas,
                                                     gt_semantic_seg,
                                                     self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses
    def _kl_divergence(self, mu,logsigma):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        losses_kl = dict()
        loss_kl = -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp()).sum() / logsigma.size(0)
        losses_kl['loss_ce'] = loss_kl*1e-5
        losses.update(add_prefix(losses_kl, 'uncertainty'))
        return losses
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
        losses_div['loss_ce'] = torch.mean(F.relu(torch.linalg.matrix_norm(ScoreMat-Identity.cuda(), dim=(-2, -1)) - self.beta[None,:]))*0.1
        losses.update(add_prefix(losses_div, 'diversity'))
        return losses

    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        return seg_logits

    def _auxiliary_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.forward_train(x, img_metas,
                                                  gt_semantic_seg,
                                                  self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.forward_train(
                x, img_metas, gt_semantic_seg, self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def _identity_head_forward_train(self, x, img_metas, gt_semantic_seg,reg , weight=None):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        loss_aux = self.identity_head.forward_train(
            x, img_metas, gt_semantic_seg, self.train_cfg,weight=weight)
        loss_aux['loss_ce'] += reg
        losses.update(add_prefix(loss_aux, 'aux_identity'))
        return losses

    def forward_dummy(self, img):
        """Dummy forward function."""
        seg_logit = self.encode_decode(img, None)

        return seg_logit

    def after_extract_feat(self, x):
        x_orig = list(x[0:4])
        global_feat, visual_embeddings = x[4]

        B, C, H, W = visual_embeddings.shape
        if self.context_feature == 'attention':
            visual_context = torch.cat([global_feat.reshape(B, C, 1), visual_embeddings.reshape(B, C, H*W)], dim=2).permute(0, 2, 1)  # B, N, C

        # (B, K, C)
        samples = []
        logsigma = []
        text_embeddings = []
        for ctx in self.contexts:
            mu_i = self.text_encoder(self.texts.to(global_feat.device), ctx).expand(B, -1, -1)
            # update text_embeddings by visual_context!
            logsigma_i = self.vcp_decoder(mu_i, visual_context)
            samples.append(self.sample_gaussian_tensors(mu_i, logsigma_i, num_samples=3))
            # samples.append(mu_i.unsqueeze(1))
            text_embeddings.append(mu_i.unsqueeze(1)) 
            logsigma.append(logsigma_i.unsqueeze(1))
        text_embeddings = torch.cat(text_embeddings, dim=1) # (B, N, C, d)
        logsigma = torch.cat(logsigma, dim=1) # (B, N, C, d)
        samples = torch.cat(samples, dim=1)
        mog_mean = torch.mean(text_embeddings, dim = 1) # (B, C, d)
        mog_var = torch.log(torch.mean(text_embeddings**2 + torch.exp(logsigma)**2, dim=1) - mog_mean**2) # (B, C, d)
        # samples = self.sample_gaussian_tensors(mog_mean, mog_var, num_samples=10) # (B, K, C, d)
        # samples = torch.cat((text_embeddings, samples), dim=1)
        
        # print(text_embeddings.size())
        # import pdb
        # pdb.set_trace()
        # compute score map and concat
        # B, M, K, C = text_embeddings.shape
        visual_embeddings = F.normalize(visual_embeddings, dim=1, p=2)
        text = F.normalize(samples, dim=-1, p=2)
        
        score_map_ = torch.einsum('bchw,bmkc->bmkhw', visual_embeddings, text)
        score_map = torch.mean(score_map_, dim=1)
        x_orig[self.score_concat_index] = torch.cat([x_orig[self.score_concat_index], score_map], dim=1)
        if self.training:
            return text_embeddings, x_orig, score_map, mog_mean, mog_var
        else: 
            return text_embeddings, x_orig, score_map_
    def forward_train(self, img, img_metas, gt_semantic_seg):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        x = self.extract_feat(img)
        _x_orig = [x[i] for i in range(4)]
        text_embeddings, x_orig, score_map, mog_mean, mog_var = self.after_extract_feat(x)
        uncert = torch.exp(torch.mean(torch.mean(mog_var, dim=-1), dim=-1)) # (B)
        scale_uncert = torch.mean(torch.log(uncert)) * 0.5
        # weight = F.one_hot(gt_semantic_seg.squeeze(1),256).permute(0,3,1,2)
        weight = torch.ones_like(gt_semantic_seg.squeeze(1)).cuda()
        # weight = None

        weight = weight*(1/uncert[:,None,None])
        # weight = weight.detach()
        # for i in len(scale_uncert.size(0)):
            
        if self.with_neck:
            x_orig = list(self.neck(x_orig))
            _x_orig = x_orig

        losses = dict()
        if self.text_head:
            x = [text_embeddings,] + x_orig
        else:
            x = x_orig

        loss_decode = self._decode_head_forward_train(x, img_metas,
                                                      gt_semantic_seg)
        losses.update(loss_decode)

        if self.with_identity_head:
            loss_identity = self._identity_head_forward_train(
                score_map/self.tau, img_metas, gt_semantic_seg,reg=scale_uncert,weight=weight)
            losses.update(loss_identity)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                _x_orig, img_metas, gt_semantic_seg)
            losses.update(loss_aux)
        
        loss_kl = self._kl_divergence(mog_mean, mog_var)
        losses.update(loss_kl)
        loss_div = self._diversity(text_embeddings)
        losses.update(loss_div)
        return losses

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(img)

        _x_orig = [x[i] for i in range(4)]
        text_embeddings, x_orig, score_map = self.after_extract_feat(x)

        if self.with_neck:
            x_orig = list(self.neck(x_orig))

        if self.text_head:
            x = [text_embeddings,] + x_orig
        else:
            x = x_orig
        # print('text_embedding=', text_embeddings[0])
        out = self._decode_head_forward_test(x, img_metas)
        # print('cls_map=', out[0,:,40, 40])
        
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out


    # TODO refactor
    def slide_inference(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.encode_decode(crop_img, img_meta)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        preds = preds / count_mat
        if rescale:
            preds = resize(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        return preds

    def whole_inference(self, img, img_meta, rescale):
        """Inference with full image."""

        seg_logit = self.encode_decode(img, img_meta)
        if rescale:
            # support dynamic shape for onnx
            if torch.onnx.is_in_onnx_export():
                size = img.shape[2:]
            else:
                size = img_meta[0]['ori_shape'][:2]
            seg_logit = resize(
                seg_logit,
                size=size,
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        
        if  torch.isnan(seg_logit).any():
            print('########### find NAN #############')

        return seg_logit

    def inference(self, img, img_meta, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(img, img_meta, rescale)
        else:
            seg_logit = self.whole_inference(img, img_meta, rescale)
        output = F.softmax(seg_logit, dim=1)
        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3, ))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2, ))

        return output

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        seg_logit = self.inference(img, img_meta, rescale)
        seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        seg_pred = list(seg_pred)
        return seg_pred
