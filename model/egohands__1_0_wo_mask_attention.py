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
from model.utils.plucker_ray import PluckerRayEmbedding


class EgoHands(nn.Module):
    def __init__(self, use_plucker_ray: bool = False, use_hand_context_mask: bool = False):
        super().__init__()

        self.use_plucker_ray = use_plucker_ray

        self.body_backbone = DinoBackbone(
            freeze_params=False,
        )
        self.hand_backbone = HaMeREncoder(
            freeze_params=True,
        )

        body_context_dim = self.body_backbone.embed_dim

        if self.use_plucker_ray:
            patch_size = self.body_backbone.patch_size
            if isinstance(patch_size, tuple):
                plucker_in_dim = 6 * patch_size[0] * patch_size[1]
            else:
                plucker_in_dim = 6 * patch_size * patch_size

            self.plucker_ray = PluckerRayEmbedding(
                image_size=(512, 512),
                patch_size=patch_size,
            )
            self.plucker_proj = nn.Sequential(
                nn.Linear(plucker_in_dim, self.body_backbone.embed_dim),
                nn.LayerNorm(self.body_backbone.embed_dim),
            )
            self.body_plucker_fuse = nn.Sequential(
                nn.Linear(self.body_backbone.embed_dim * 2, self.body_backbone.embed_dim),
                nn.LayerNorm(self.body_backbone.embed_dim),
            )

        self.decoder = BaseDecoder(
            input_dim=body_context_dim,
            use_hand_context_mask=use_hand_context_mask,
        )


        self.mano_lhand = MANOLayer(model_path=MANO_PATH, is_rhand=False)
        self.mano_rhand = MANOLayer(model_path=MANO_PATH, is_rhand=True)

        for param in self.mano_lhand.parameters():
            param.requires_grad = False
        for param in self.mano_rhand.parameters():
            param.requires_grad = False

    @staticmethod
    def _ensure_temporal_dim(x):
        if x.dim() == 4:
            return x.unsqueeze(1), False
        return x, True
    
    def _build_plucker_tokens(self, batch, batch_size, seq_len, device, dtype):
        intrinsics = batch.get("intrinsics_resize", batch["intrinsics"]).to(device=device, dtype=dtype)

        if intrinsics.dim() == 3:
            intrinsics = intrinsics.unsqueeze(1)
        if intrinsics.dim() == 4 and intrinsics.shape[2] == 1:
            intrinsics = intrinsics.squeeze(2)

        intrinsics = intrinsics.reshape(batch_size, seq_len, 4)

        # Use identity extrinsics so Plucker embedding depends only on intrinsics.
        extrinsics_4x4 = (
            torch.eye(4, device=device, dtype=dtype)
            .view(1, 1, 4, 4)
            .expand(batch_size, seq_len, 4, 4)
            .clone()
        )

        plucker_tokens = self.plucker_ray(
            extrinsics=extrinsics_4x4,
            intrinsics=intrinsics,
        )
        plucker_tokens = plucker_tokens.reshape(batch_size * seq_len, plucker_tokens.shape[2], plucker_tokens.shape[3])
        plucker_tokens = self.plucker_proj(plucker_tokens)

        return plucker_tokens

    def forward(self, batch, weight_dtype, return_intermediate=False):
        images = batch["image"].to(weight_dtype)
        left_hand_images = batch["left_hand_image"].to(weight_dtype)
        right_hand_images = batch["right_hand_image"].to(weight_dtype)
        betas_gt = batch["betas"].to(weight_dtype)

        images, has_temporal = self._ensure_temporal_dim(images)
        left_hand_images, _ = self._ensure_temporal_dim(left_hand_images)
        right_hand_images, _ = self._ensure_temporal_dim(right_hand_images)

        # betas are [B, 2, 10] (single frame) or [B, T, 2, 10] (temporal);
        # only single-frame inputs need a temporal dimension.
        if betas_gt.dim() == 3:
            betas_gt = betas_gt.unsqueeze(1)

        batch_size, seq_len, _, _, _ = images.shape

        images_bt = images.reshape(batch_size * seq_len, *images.shape[2:])
        left_hand_images_bt = left_hand_images.reshape(batch_size * seq_len, *left_hand_images.shape[2:])
        right_hand_images_bt = right_hand_images.reshape(batch_size * seq_len, *right_hand_images.shape[2:])
        betas_bt = betas_gt.reshape(batch_size * seq_len, *betas_gt.shape[2:])

        body_feat = self.body_backbone(images_bt)
        bt, feat_dim, feat_height, feat_width = body_feat.shape
        body_tokens = body_feat.permute(0, 2, 3, 1).reshape(bt, feat_height * feat_width, feat_dim)

        if self.use_plucker_ray:
            plucker_tokens = self._build_plucker_tokens(
                batch=batch,
                batch_size=batch_size,
                seq_len=seq_len,
                device=body_tokens.device,
                dtype=body_tokens.dtype,
            )
            # Concatenate first, then fuse back to decoder context dimension.
            body_tokens = torch.cat([body_tokens, plucker_tokens], dim=-1)
            body_tokens = self.body_plucker_fuse(body_tokens)

        hand_images = torch.stack([left_hand_images_bt, right_hand_images_bt], dim=1)
        hand_feat = self.hand_backbone(hand_images) 
        bt, num_side, feat_dim, feat_height, feat_width = hand_feat.shape
        hand_tokens = hand_feat.permute(0, 1, 3, 4, 2).reshape(bt, num_side, feat_height * feat_width, feat_dim)

        outputs_bt = self.decoder(
            hand_tokens,
            body_tokens,
            return_intermediate=return_intermediate,
        )
        pose_pred_bt = outputs_bt["pose"]
        transl_pred_bt = outputs_bt["transl"]
        mano_tokens_bt = outputs_bt.get("mano_tokens")

        mano_outputs_left = self.mano_lhand(
            pose_type="rot6d",
            betas=betas_bt[:, 0],
            global_orient=pose_pred_bt[:, 0, :1],
            hand_pose=pose_pred_bt[:, 0, 1:],
            transl=transl_pred_bt[:, 0],
        )
        mano_outputs_right = self.mano_rhand(
            pose_type="rot6d",
            betas=betas_bt[:, 1],
            global_orient=pose_pred_bt[:, 1, :1],
            hand_pose=pose_pred_bt[:, 1, 1:],
            transl=transl_pred_bt[:, 1],
        )

        joints3d_pred_bt = torch.stack(
            [mano_outputs_left["joints"], mano_outputs_right["joints"]],
            dim=1,
        )

        pose_pred = pose_pred_bt.reshape(batch_size, seq_len, 2, 16, 6)
        transl_pred = transl_pred_bt.reshape(batch_size, seq_len, 2, 3)
        joints3d_pred = joints3d_pred_bt.reshape(batch_size, seq_len, 2, 21, 3)

        if not has_temporal:
            pose_pred = pose_pred[:, 0]
            transl_pred = transl_pred[:, 0]
            joints3d_pred = joints3d_pred[:, 0]
            if mano_tokens_bt is not None:
                mano_tokens = mano_tokens_bt
        else:
            if mano_tokens_bt is not None:
                mano_tokens = mano_tokens_bt.reshape(batch_size, seq_len, *mano_tokens_bt.shape[1:])

        outputs = {
            "pose": pose_pred,
            "transl": transl_pred,
            "joints3d": joints3d_pred,
        }

        # Add intermediate features for knowledge distillation
        if return_intermediate:
            outputs["body_tokens"] = body_tokens
            outputs["hand_tokens"] = hand_tokens
            if mano_tokens_bt is not None:
                outputs["mano_tokens"] = mano_tokens if has_temporal else mano_tokens_bt

        losses = self.compute_loss(outputs, batch, weight_dtype)
        metrics = self.compute_metric(outputs, batch, weight_dtype)

        return losses, metrics, outputs

    def compute_loss(self, outputs, batch, weight_dtype, eps=1e-8):
        pose = batch["pose"].to(weight_dtype)
        transl = batch["transl"].to(weight_dtype)

        joints3d = batch["joints3d"].to(weight_dtype)

        hand_side_mask = batch["hand_side_mask"].to(weight_dtype)
        mano_side_mask = batch.get("mano_side_mask", batch["hand_side_mask"]).to(weight_dtype)

        pose_pred = outputs["pose"]
        transl_pred = outputs["transl"]
        joints3d_pred = outputs["joints3d"]
        
        pose_loss = F.l1_loss(pose_pred, pose, reduction="none")
        pose_loss= pose_loss.mean(dim=-1).mean(dim=-1)
        pose_loss = (pose_loss * mano_side_mask).sum() / (mano_side_mask.sum() + eps)

        transl_loss = F.l1_loss(transl_pred, transl, reduction="none")
        transl_loss= transl_loss.mean(dim=-1)
        transl_loss = (transl_loss * mano_side_mask).sum() / (mano_side_mask.sum() + eps)

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

        if hand_side_mask.dim() == 2:
            joints3d_eval = joints3d.unsqueeze(1)
            joints3d_pred_eval = joints3d_pred.unsqueeze(1)
            hand_side_mask_eval = hand_side_mask.unsqueeze(1)
        else:
            joints3d_eval = joints3d
            joints3d_pred_eval = joints3d_pred
            hand_side_mask_eval = hand_side_mask

        eval_batch = joints3d_eval.shape[0] * joints3d_eval.shape[1]
        joints3d_eval = joints3d_eval.reshape(eval_batch, 2, 21, 3)
        joints3d_pred_eval = joints3d_pred_eval.reshape(eval_batch, 2, 21, 3)
        hand_side_mask_eval = hand_side_mask_eval.reshape(eval_batch, 2)

        batch_size, num_hands = hand_side_mask_eval.shape
        for batch_idx in range(batch_size):
            for hand_idx in range(num_hands):
                if hand_side_mask_eval[batch_idx, hand_idx] <= 0:
                    continue
                pred_aligned = self._align_w_scale(
                    joints3d_eval[batch_idx, hand_idx],
                    joints3d_pred_eval[batch_idx, hand_idx],
                )
                pa_mpjpe_sum = pa_mpjpe_sum + torch.norm(
                    pred_aligned - joints3d_eval[batch_idx, hand_idx],
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
