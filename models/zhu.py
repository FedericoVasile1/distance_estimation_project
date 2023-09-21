import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce

from trainers.base_trainer import Trainer
from models.backbones import BACKBONES
from models.base_model import BaseLifter
from models.bb_regressors import REGRESSORS
from models.losses.gaussian_nll import GNLL
from models.losses.laplacian_nll import LNLL
from models.losses.smooth_l1 import SmoothL1
from models.losses.zhu_enhanced import EnhancedZHU
from trainers.trainer_regressor import TrainerRegressor


class ZHU(BaseLifter):
    TEMPORAL = False

    def __init__(self, args: argparse.Namespace, enhanced: bool = False):
        super().__init__()
        args.use_centers = False
        self.enhanced = enhanced
        self.alpha = args.alpha_zhu
        self.loss = args.loss

        assert not (self.enhanced and self.loss in ('gaussian', 'laplacian')),\
            "Enhanced ZHU is not compatible with gaussian or laplacian loss"

        self.output_size = 2 if self.loss in ('gaussian', 'laplacian') else 1

        self.backbone = BACKBONES[args.backbone](args)
        self.regressor = REGRESSORS[args.regressor](input_dim=self.backbone.output_size,
                                                    pool_size=2,
                                                    roi_op=args.roi_op,)

        # args.freeze_at == 0 means all unfreezed
        # args.freeze_at == 1 means freeze up to first layer (included)
        # etc..
        
        # TODO should be moved in backbone
        for layer_idx in range(1, args.freeze_at + 1):
            layer = getattr(self.backbone.resnet, f'layer{layer_idx}')
            for param in layer.parameters():
                param.requires_grad = False

        feat_vect_size = self.backbone.output_size * 2 * 2
        if args.img_featmap:
            feat_vect_size += self.backbone.output_size
            self.img_featmap = True
        else:
            self.img_featmap = False

        if args.rnn:
            self.rnn_hidden_size = args.rnn_hidden_size
            self.rnn = nn.LSTMCell(
                feat_vect_size, self.rnn_hidden_size
            )
            self.dropout = nn.Dropout(p=0.1)
            self.classifier = nn.Linear(self.rnn_hidden_size, self.output_size)
        else:
            self.rnn = None

            self.distance_estimator = nn.Sequential(
                nn.Identity() if self.img_featmap else nn.Flatten(),
                nn.Linear(feat_vect_size, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, self.output_size),
            )

        if self.enhanced:
            self.keypoint_regressor = nn.Sequential(
                nn.Identity() if self.img_featmap else nn.Flatten(),
                nn.Linear(feat_vect_size, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 2),
                nn.Tanh())

    def forward(self, x: torch.Tensor, bboxes: torch.Tensor) -> torch.Tensor:
        W = x.shape[-1]
        B = x.shape[0]
        x = rearrange(x, 'b c 1 h w -> b c h w')
        x = self.backbone(x)

        if self.img_featmap:
            x_gblavg = reduce(x, 'b c h w -> b c', reduction="mean")
            x_gblavg_bboxes = torch.repeat_interleave(
                x_gblavg,
                torch.tensor([len(b) for b in bboxes]).to(x.device),
                dim=0
            )

        x = self.regressor(x, bboxes, scale=x.shape[-1] / W)

        if self.rnn is None:
            if self.img_featmap:
                x = x.flatten(start_dim=1)
                x = torch.cat([x, x_gblavg_bboxes], dim=-1)

            z = self.distance_estimator(x).squeeze(-1)
        else:
            # NOTE: every bboxes elem (i.e. the bounding boxes in the image)
            #       is assumed to be sorted by bounding box height

            # 1. pad roi with zeros up to maximum num of boxes in image
            max_img_bboxes = -1
            for img_bboxes in bboxes:
                if img_bboxes.shape[0] > max_img_bboxes:
                    max_img_bboxes = img_bboxes.shape[0]
            feat_map_shape = x.shape[1:]
            x_padded = torch.zeros(B, max_img_bboxes, *feat_map_shape).to(x.device, x.dtype)

            pointer = 0
            for idx in range(len(bboxes)):
                n_img_bboxes = bboxes[idx].shape[0]
                x_padded[idx, :n_img_bboxes] = x[pointer:pointer + n_img_bboxes]
                pointer += n_img_bboxes
            x = x_padded

            # x.shape (bsize, max_img_bboxes, CC, HH, WW)
            
            # 2. forward boxes in rnn step by step
            b_size, n_steps = x.shape[:2]
            h_n = torch.zeros(b_size, self.rnn_hidden_size).to(x.device, x.dtype)
            c_n = torch.zeros_like(h_n).to(device=x.device, dtype=x.dtype)
            z = torch.zeros(b_size, n_steps).to(x.device, x.dtype)
            for step in range(x.shape[1]):
                x_step = x[:, step].flatten(start_dim=1)
                if self.img_featmap:
                    x_step = torch.cat([x_step, x_gblavg], dim=-1)
                
                h_n, c_n = self.rnn(self.dropout(x_step), (h_n, c_n)) 
                temp = self.classifier(self.dropout(h_n)).squeeze()
                z[:, step] = temp
            z = z.reshape(-1)

        if self.loss in ('gaussian', 'laplacian'):
            mu = F.softplus(z[..., 0])
            logvar = z[..., 1]
            z = (mu, logvar)
        else:
            z = F.softplus(z)

        if self.enhanced and self.training:
            k = self.keypoint_regressor(x)
            return z, k

        return z

    def get_loss_fun(self, **kwargs) -> nn.Module:
        if self.enhanced:
            return EnhancedZHU(alpha=self.alpha, train=self.training)
        return {'l1': SmoothL1, 'gaussian': GNLL, 'laplacian': LNLL}[self.loss]()

    def get_trainer(self) -> Trainer:
        return TrainerRegressor


if __name__ == '__main__':
    from main import parse_args
    args = parse_args()
    args.backbone = 'vgg16'

    model = ZHU(args).to(args.device)
    x = torch.rand((args.batch_size, 3, 1, 512, 1024)).to(args.device)
    y = model.forward(x)

    print(f'$> RESNET-{model.depth}')
    print(f'───$> input shape: {tuple(x.shape)}')
    print(f'───$> output shape: {tuple(y.shape)}')
