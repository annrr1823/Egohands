# Edited from https://github.com/Haian-Jin/LVSM/blob/main/utils/data_utils.py

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PluckerRayEmbedding(nn.Module):
    def __init__(
        self,
        image_size=(256, 192),
        patch_size=16,
    ):
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size

        height, width = self.image_size
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError(f"The image_size {self.image_size} must be divisible by patch_size {self.patch_size}.")


    @torch.no_grad()
    def compute_plucker_ray(self, extrinsics, intrinsics):
        """
        Args:
            extrinsics (T_camera_to_world): [B, V, 4, 4]
            intrinsics ([fx, fy, cx, cy]): [B, V, 4]
        Returns:
            plucker_ray: [B, V, 6, H, W]
        """

        batch_size, num_views = extrinsics.shape[:2]
        device = extrinsics.device
        dtype = extrinsics.dtype

        extrinsics = extrinsics.reshape(batch_size * num_views, 4, 4)
        intrinsics = intrinsics.reshape(batch_size * num_views, 4).to(device=device, dtype=dtype)

        height, width = self.image_size

        y, x = torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype),
            torch.arange(width, device=device, dtype=dtype),
            indexing="ij",
        )

        x = x.reshape(1, -1).expand(batch_size * num_views, -1)
        y = y.reshape(1, -1).expand(batch_size * num_views, -1)

        x = (x + 0.5 - intrinsics[:, 2:3]) / intrinsics[:, 0:1]
        y = (y + 0.5 - intrinsics[:, 3:4]) / intrinsics[:, 1:2]
        z = torch.ones_like(x)

        ray_d = torch.stack([x, y, z], dim=-1)
        ray_d = torch.bmm(ray_d, extrinsics[:, :3, :3].transpose(1, 2))
        ray_d = ray_d / torch.linalg.norm(ray_d, dim=-1, keepdim=True).clamp_min(1e-8)

        ray_o = extrinsics[:, :3, 3][:, None, :].expand_as(ray_d)
        o_cross_d = torch.cross(ray_o, ray_d, dim=-1)
        plucker = torch.cat([o_cross_d, ray_d], dim=-1)

        plucker = plucker.view(batch_size, num_views, height, width, 6)
        plucker = plucker.permute(0, 1, 4, 2, 3).contiguous()

        return plucker


    @torch.no_grad()
    def patchify_plucker_ray(self, plucker_ray):
        """
        Args:
            plucker_ray: [B, V, 6, H, W]
        Returns:
            plucker_tokens: [B, V, N_patch, 6 * patch_size * patch_size]
        """

        batch_size, num_views, channels, height, width = plucker_ray.shape
        patch_size = self.patch_size

        if height != self.image_size[0] or width != self.image_size[1]:
            raise ValueError(
                f"Expected cropped size {self.image_size}, got {(height, width)}."
            )

        num_patch_h = height // patch_size
        num_patch_w = width // patch_size

        plucker_tokens = plucker_ray.view(
            batch_size,
            num_views,
            channels,
            num_patch_h,
            patch_size,
            num_patch_w,
            patch_size,
        )
        plucker_tokens = plucker_tokens.permute(0, 1, 3, 5, 2, 4, 6).contiguous()
        plucker_tokens = plucker_tokens.view(
            batch_size,
            num_views,
            num_patch_h * num_patch_w,
            channels * patch_size * patch_size,
        )

        return plucker_tokens

    @torch.no_grad()
    def compute_patch_center_plucker(self, extrinsics, intrinsics):
        """
        Compute one Plucker ray per patch center.

        Args:
            extrinsics: [B, V, 4, 4] camera-to-world matrices.
            intrinsics: [B, V, 4] with [fx, fy, cx, cy].

        Returns:
            plucker_tokens: [B, V, N_patch, 6]
        """

        batch_size, num_views = extrinsics.shape[:2]
        device = extrinsics.device
        dtype = extrinsics.dtype

        extrinsics = extrinsics.reshape(batch_size * num_views, 4, 4)
        intrinsics = intrinsics.reshape(batch_size * num_views, 4).to(device=device, dtype=dtype)

        height, width = self.image_size
        patch_size = self.patch_size
        num_patch_h = height // patch_size
        num_patch_w = width // patch_size

        patch_y, patch_x = torch.meshgrid(
            torch.arange(num_patch_h, device=device, dtype=dtype),
            torch.arange(num_patch_w, device=device, dtype=dtype),
            indexing="ij",
        )
        patch_x = patch_x.reshape(1, -1).expand(batch_size * num_views, -1)
        patch_y = patch_y.reshape(1, -1).expand(batch_size * num_views, -1)

        x = (patch_x + 0.5) * patch_size
        y = (patch_y + 0.5) * patch_size

        x = (x - intrinsics[:, 2:3]) / intrinsics[:, 0:1]
        y = (y - intrinsics[:, 3:4]) / intrinsics[:, 1:2]
        z = torch.ones_like(x)

        ray_d = torch.stack([x, y, z], dim=-1)
        ray_d = torch.bmm(ray_d, extrinsics[:, :3, :3].transpose(1, 2))
        ray_d = ray_d / torch.linalg.norm(ray_d, dim=-1, keepdim=True).clamp_min(1e-8)

        ray_o = extrinsics[:, :3, 3][:, None, :].expand_as(ray_d)
        o_cross_d = torch.cross(ray_o, ray_d, dim=-1)
        plucker = torch.cat([o_cross_d, ray_d], dim=-1)

        plucker = plucker.view(batch_size, num_views, num_patch_h * num_patch_w, 6)
        return plucker

    def forward(self, extrinsics, intrinsics):
        plucker_ray = self.compute_plucker_ray(
            extrinsics=extrinsics,
            intrinsics=intrinsics
        )
        plucker_tokens = self.patchify_plucker_ray(plucker_ray)

        return plucker_tokens
