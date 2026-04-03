# Edited from https://github.com/facebookresearch/sam-3d-body/blob/main/sam_3d_body/models/backbones/dinov3.py

from pathlib import Path
import sys
import torch
from torch import nn

DEFAULT_SAM3DB_CKPT = "/data/sunzhihao/egohands/third_party/ckpt/model_dino.ckpt"


class DinoBackbone(nn.Module):
    def __init__(
        self,
        freeze_params=False,
        sam3db_pretrained=DEFAULT_SAM3DB_CKPT,
    ):
        super().__init__()

        self.freeze_params = freeze_params
        self.sam3db_pretrained = sam3db_pretrained

        self.name = "dinov3_vith16plus"
        self.repo = Path(__file__).parent.parent.parent / "third_party" / "dinov3"
        if str(self.repo) not in sys.path:
            sys.path.insert(0, str(self.repo))

        self.model = torch.hub.load(
            self.repo,
            self.name,
            source="local",
            pretrained=False,
        )
        self.patch_size = self.model.patch_size
        self.embed_dim = self.embed_dims = self.model.embed_dim

        if sam3db_pretrained is not None:
            state_dict = torch.load(sam3db_pretrained, map_location="cpu", weights_only=False)

            backbone_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("backbone.encoder."):
                    _k = k.replace("backbone.encoder.", "")
                    backbone_state_dict[_k] = v

            missing, unexpected = self.model.load_state_dict(backbone_state_dict, strict=False)

            if missing or unexpected:
                print(
                    f"Loaded SAM3D Body DINOv3 backbone with missing keys: {missing}, "
                    f"unexpected keys: {unexpected}"
                )

        if freeze_params:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

            
    def forward(self, x, extra_embed=None):
        """
        Encode a RGB image using a ViT-backbone
        Args:
            - x: torch.Tensor of shape [bs, 3, w, h]
        Return:
            - y: torch.Tensor of shape [bs, k, d] - image in patchified mode
        """
        assert extra_embed is None, "Not Implemented Yet"

        y = self.model.get_intermediate_layers(x, n=1, reshape=True, norm=True)[-1]

        return y

    def get_layer_depth(self, param_name: str, prefix: str = "encoder."):
        """Get the layer-wise depth of a parameter.
        Args:
            param_name (str): The name of the parameter.
            prefix (str): The prefix for the parameter.
                Defaults to an empty string.
        Returns:
            Tuple[int, int]: The layer-wise depth and the num of layers.
        Note:
            The first depth is the stem module (``layer_depth=0``), and the
            last depth is the subsequent module (``layer_depth=num_layers-1``)
        """
        num_layers = self.model.n_blocks + 2

        if not param_name.startswith(prefix):
            # For subsequent module like head
            return num_layers - 1, num_layers

        param_name = param_name[len(prefix) :]

        if param_name in ("cls_token", "pos_embed", "storage_tokens"):
            layer_depth = 0
        elif param_name.startswith("patch_embed"):
            layer_depth = 0
        elif param_name.startswith("blocks"):
            layer_id = int(param_name.split(".")[1])
            layer_depth = layer_id + 1
        else:
            layer_depth = num_layers - 1

        return layer_depth, num_layers


if __name__ == "__main__":
    backbone = DinoBackbone()
