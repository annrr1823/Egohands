import numpy as np
import torch
import torch.nn as nn

from model.component.transformer import DecoderDoubleCrossAttn, SelfAttention

DEFAULT_HAMER_CKPT = "/data/sunzhihao2/turuirui/dataset/hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt"
DEFAULT_MANO_MEAN_PARAMS_PATH = "/data/sunzhihao2/turuirui/dataset/hamer/_DATA/data/mano_mean_params.npz"


class BaseDecoder(nn.Module):
    def __init__(
        self, 
        input_dim=1280,
        use_temporal_self_attn: bool = True,
        max_temporal_len: int = 32,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.use_temporal_self_attn = use_temporal_self_attn
        self.max_temporal_len = max_temporal_len

        self.hidden_dim = 1024
        self.hamer_ckpt = DEFAULT_HAMER_CKPT
        self.mean_params_path = DEFAULT_MANO_MEAN_PARAMS_PATH

        self.pose_dim = 16 * 6
        self.transl_dim = 3
        self.joints3d_dim = 21 * 3

        mean_params = np.load(self.mean_params_path)
        init_pose = torch.from_numpy(mean_params["pose"].astype(np.float32)).unsqueeze(0)
        self.register_buffer("init_pose", init_pose)

        self.left_pose_token_proj = nn.Linear(self.pose_dim, self.hidden_dim)
        self.right_pose_token_proj = nn.Linear(self.pose_dim, self.hidden_dim)

        self.left_transl_token = nn.Parameter(torch.randn(self.hidden_dim) * 0.02)
        self.right_transl_token = nn.Parameter(torch.randn(self.hidden_dim) * 0.02)

        self.decoder = DecoderDoubleCrossAttn(
            num_tokens=4,
            dim=self.hidden_dim,
            depth=6,
            heads=8,
            mlp_dim=self.hidden_dim,
            dim_head=64,
            dropout=0.0,
            emb_dropout=0.0,
            norm="layer",
            context_dim=input_dim,
        )

        if self.use_temporal_self_attn:
            self.temporal_pos_embedding = nn.Parameter(
                torch.randn(1, max_temporal_len, self.hidden_dim) * 0.02
            )
            self.temporal_self_attn = SelfAttention(
                self.hidden_dim,
                heads=8,
                dim_head=64,
                dropout=0.0,
            )
            self.temporal_norm = nn.LayerNorm(self.hidden_dim)
            self.temporal_alpha = nn.Parameter(torch.tensor(1.0))

        self._load_pretrained_decoder(self.hamer_ckpt)

        self.pose_head = nn.Linear(self.hidden_dim, self.pose_dim)
        self.transl_head = nn.Linear(self.hidden_dim, self.transl_dim)

    def _apply_temporal_self_attention(self, tokens, batch_size, seq_len):
        if not self.use_temporal_self_attn or seq_len <= 1:
            return tokens

        if seq_len > self.max_temporal_len:
            raise ValueError(
                f"seq_len={seq_len} exceeds max_temporal_len={self.max_temporal_len}"
            )

        if tokens.shape[0] != batch_size * seq_len:
            raise ValueError(
                f"Expected tokens with batch size {batch_size * seq_len}, got {tokens.shape[0]}"
            )

        tokens = tokens.reshape(batch_size, seq_len, 4, self.hidden_dim)
        temporal_pos = self.temporal_pos_embedding[:, :seq_len].unsqueeze(2)
        tokens = tokens + temporal_pos

        tokens = tokens.permute(0, 2, 1, 3).reshape(batch_size * 4, seq_len, self.hidden_dim)
        residual = tokens
        tokens = self.temporal_norm(tokens)
        tokens = self.temporal_self_attn(tokens)
        tokens = residual + self.temporal_alpha * tokens

        tokens = tokens.reshape(batch_size, 4, seq_len, self.hidden_dim).permute(0, 2, 1, 3)
        return tokens.reshape(batch_size * seq_len, 4, self.hidden_dim)

    def _load_pretrained_decoder(self, hamer_ckpt):
        checkpoint = torch.load(hamer_ckpt, map_location="cpu")
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        _state_dict = {
            key[len("mano_head.transformer."):]: value
            for key, value in state_dict.items()
            if key.startswith("mano_head.transformer.")
        }

        decoder_state_dict = {}
        for key, value in _state_dict.items():
            if key == "pos_embedding":
                continue
            elif key.startswith("transformer.layers."):
                parts = key.split(".")
                layer_idx = parts[2]
                block_idx = parts[3]
                suffix = ".".join(parts[4:])
                if block_idx == "0":
                    decoder_state_dict[key] = value
                elif block_idx == "1":
                    decoder_state_dict[f"transformer.layers.{layer_idx}.1.{suffix}"] = value
                    decoder_state_dict[f"transformer.layers.{layer_idx}.2.{suffix}"] = value.clone()
                elif block_idx == "2":
                    decoder_state_dict[f"transformer.layers.{layer_idx}.3.{suffix}"] = value
            else:
                decoder_state_dict[key] = value

        missing, unexpected = self.decoder.load_state_dict(decoder_state_dict, strict=False)

        if missing or unexpected:
            print(
                f"Loaded HaMeR decoder with missing keys: {missing}, "
                f"unexpected keys: {unexpected}"
            )

    def forward(self, hand_tokens, body_tokens, batch_size=None, seq_len=None):
        flat_batch_size = body_tokens.shape[0]
        temporal_batch_size = batch_size
        if temporal_batch_size is None and seq_len is not None:
            if flat_batch_size % seq_len != 0:
                raise ValueError(
                    f"Cannot infer temporal batch size from flat_batch_size={flat_batch_size} and seq_len={seq_len}"
                )
            temporal_batch_size = flat_batch_size // seq_len

        left_init_pose = self.init_pose.expand(flat_batch_size, -1)
        right_init_pose = self.init_pose.expand(flat_batch_size, -1)

        left_pose_token = self.left_pose_token_proj(left_init_pose)
        left_transl_token = self.left_transl_token.unsqueeze(0).expand(flat_batch_size, -1)
        right_pose_token = self.right_pose_token_proj(right_init_pose)
        right_transl_token = self.right_transl_token.unsqueeze(0).expand(flat_batch_size, -1)

        tokens = torch.stack([left_pose_token, left_transl_token, right_pose_token, right_transl_token], dim=1)

        num_left_tokens = hand_tokens.shape[2]
        hand_tokens = hand_tokens.reshape(flat_batch_size, -1, hand_tokens.shape[-1])

        hand_context_mask = torch.ones(flat_batch_size, 4, hand_tokens.shape[1], dtype=torch.bool, device=hand_tokens.device)
        hand_context_mask[:, 0, :num_left_tokens] = False
        hand_context_mask[:, 1, :num_left_tokens] = False
        hand_context_mask[:, 2, num_left_tokens:] = False
        hand_context_mask[:, 3, num_left_tokens:] = False

        tokens = self.decoder(
            tokens,
            context1=hand_tokens,
            context1_mask=hand_context_mask,
            context2=body_tokens,
            context2_mask=None,
        )

        if temporal_batch_size is not None and seq_len is not None:
            tokens = self._apply_temporal_self_attention(tokens, temporal_batch_size, seq_len)

        left_pose_pred = left_init_pose + self.pose_head(tokens[:, 0])
        left_transl_pred = self.transl_head(tokens[:, 1])
        right_pose_pred = right_init_pose + self.pose_head(tokens[:, 2])
        right_transl_pred = self.transl_head(tokens[:, 3])

        pose_pred = torch.stack([left_pose_pred, right_pose_pred], dim=1).view(flat_batch_size, 2, 16, 6)
        transl_pred = torch.stack([left_transl_pred, right_transl_pred], dim=1).view(flat_batch_size, 2, 3)

        return {
            "pose": pose_pred,
            "transl": transl_pred,
        }
