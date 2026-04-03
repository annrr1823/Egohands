import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["README.md"],
    pythonpath=True,
    dotenv=True,
)

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbone.dinov3 import DinoBackbone
from model.backbone.hamer import HaMeREncoder
from model.decoder.base_decoder import BaseDecoder
from model.smpl.mano import MANOLayer, MANO_PATH


class EgoHands(nn.Module):
    def __init__(self):
        super().__init__()

        self.body_backbone = DinoBackbone(
            freeze_params=False,
        )
        self.hand_backbone = HaMeREncoder(
            freeze_params=True,
        )

        self.decoder = BaseDecoder(
            input_dim=1280,
        )

        self.mano_lhand = MANOLayer(model_path=MANO_PATH, is_rhand=False)
        self.mano_rhand = MANOLayer(model_path=MANO_PATH, is_rhand=True)

        for param in self.mano_lhand.parameters():
            param.requires_grad = False
        for param in self.mano_rhand.parameters():
            param.requires_grad = False

    def forward(self, batch, weight_dtype):
        images = batch["image"].to(weight_dtype)
        left_hand_images = batch["left_hand_image"].to(weight_dtype)
        right_hand_images = batch["right_hand_image"].to(weight_dtype)

        betas_gt = batch["betas"].to(weight_dtype)

        body_feat = self.body_backbone(images)
        batch_size, feat_dim, feat_height, feat_width = body_feat.shape
        body_tokens = body_feat.permute(0, 2, 3, 1).reshape(batch_size, feat_height * feat_width, feat_dim)

        hand_images = torch.stack([left_hand_images, right_hand_images], dim=1)
        hand_feat = self.hand_backbone(hand_images)
        batch_size, num_side, feat_dim, feat_height, feat_width = hand_feat.shape
        hand_tokens = hand_feat.permute(0, 1, 3, 4, 2).reshape(batch_size, num_side, feat_height * feat_width, feat_dim)

        outputs = self.decoder(hand_tokens, body_tokens)

        pose_pred = outputs["pose"]
        transl_pred = outputs["transl"]

        mano_outputs_left = self.mano_lhand(
            pose_type="rot6d",
            betas=betas_gt[:, 0],
            global_orient=pose_pred[:, 0, :1],
            hand_pose=pose_pred[:, 0, 1:],
            transl=transl_pred[:, 0],
        )
        mano_outputs_right = self.mano_rhand(
            pose_type="rot6d",
            betas=betas_gt[:, 1],
            global_orient=pose_pred[:, 1, :1],
            hand_pose=pose_pred[:, 1, 1:],
            transl=transl_pred[:, 1],
        )

        joints3d_pred = torch.stack(
            [mano_outputs_left["joints"], mano_outputs_right["joints"]],
            dim=1,
        )

        outputs = {
            "pose": pose_pred,
            "transl": transl_pred,
            "joints3d": joints3d_pred,
        }

        losses = self.compute_loss(outputs, batch, weight_dtype)
        metrics = self.compute_metric(outputs, batch, weight_dtype)

        return losses, metrics, outputs

    def compute_loss(self, outputs, batch, weight_dtype, eps=1e-8):
        pose = batch["pose"].to(weight_dtype)
        transl = batch["transl"].to(weight_dtype)

        joints3d = batch["joints3d"].to(weight_dtype)

        hand_side_mask = batch["hand_side_mask"].to(weight_dtype)

        pose_pred = outputs["pose"]
        transl_pred = outputs["transl"]
        joints3d_pred = outputs["joints3d"]
        
        pose_loss = F.l1_loss(pose_pred, pose, reduction="none")
        pose_loss= pose_loss.mean(dim=-1).mean(dim=-1)
        pose_loss = (pose_loss * hand_side_mask).sum() / (hand_side_mask.sum() + eps)

        transl_loss = F.l1_loss(transl_pred, transl, reduction="none")
        transl_loss= transl_loss.mean(dim=-1)
        transl_loss = (transl_loss * hand_side_mask).sum() / (hand_side_mask.sum() + eps)

        joints3d_loss = F.l1_loss(joints3d_pred, joints3d, reduction="none")
        joints3d_loss= joints3d_loss.mean(dim=-1).mean(dim=-1)
        joints3d_loss = (joints3d_loss * hand_side_mask).sum() / (hand_side_mask.sum() + eps)

        losses = {
            "pose": pose_loss,
            "transl": transl_loss,
            "joints3d": joints3d_loss,
        }

        return losses

    @staticmethod
    def _align_w_scale(gt_points, pred_points):
        t_gt = gt_points.mean(dim=0, keepdim=True)
        t_pred = pred_points.mean(dim=0, keepdim=True)

        gt_centered = gt_points - t_gt
        pred_centered = pred_points - t_pred

        s_gt = torch.linalg.norm(gt_centered) + 1e-8
        s_pred = torch.linalg.norm(pred_centered) + 1e-8
        gt_centered = gt_centered / s_gt
        pred_centered = pred_centered / s_pred

        covariance = pred_centered.transpose(0, 1) @ gt_centered
        u, s, v_t = torch.linalg.svd(covariance)
        r = v_t.transpose(0, 1) @ u.transpose(0, 1)
        if torch.det(r) < 0:
            v_t = v_t.clone()
            v_t[-1, :] *= -1
            r = v_t.transpose(0, 1) @ u.transpose(0, 1)
        scale = s.sum()

        pred_aligned = (pred_centered @ r.transpose(0, 1)) * scale
        pred_aligned = pred_aligned * s_gt + t_gt
        
        return pred_aligned

    def compute_metric(self, outputs, batch, weight_dtype, eps=1e-8):
        joints3d = batch["joints3d"].to(weight_dtype)
        hand_side_mask = batch["hand_side_mask"].to(weight_dtype)

        joints3d_pred = outputs["joints3d"]

        mpjpe = torch.norm(joints3d_pred - joints3d, dim=-1)
        mpjpe = mpjpe.mean(dim=-1)
        mpjpe = (mpjpe * hand_side_mask).sum() / (hand_side_mask.sum() + eps)

        pa_mpjpe_sum = joints3d_pred.new_zeros(())
        batch_size, num_hands = hand_side_mask.shape
        for batch_idx in range(batch_size):
            for hand_idx in range(num_hands):
                if hand_side_mask[batch_idx, hand_idx] <= 0:
                    continue
                pred_aligned = self._align_w_scale(
                    joints3d[batch_idx, hand_idx],
                    joints3d_pred[batch_idx, hand_idx],
                )
                pa_mpjpe_sum = pa_mpjpe_sum + torch.norm(
                    pred_aligned - joints3d[batch_idx, hand_idx],
                    dim=-1,
                ).mean()
        pa_mpjpe = pa_mpjpe_sum / (hand_side_mask.sum() + eps)

        metrics = {
            "mpjpe": mpjpe,
            "pa_mpjpe": pa_mpjpe,
        }

        return metrics
    

    def train(self, mode: bool = True):
        super().train(mode)

        if self.body_backbone.freeze_params:
            self.body_backbone.eval()
        if self.hand_backbone.freeze_params:
            self.hand_backbone.eval()
            
        return self


