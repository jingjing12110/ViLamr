import torch
import torch.nn as nn

import open_clip


class CLIPCNN(nn.Module):
    def __init__(
            self,
            model_name='convnext_xxlarge',
            delay_load=True,
            model_path="playground/checkpoints/openai/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-soup"
    ):
        super().__init__()
        self.is_loaded = False

        self.vision_tower_name = model_name
        self.model_path = model_path
        self.visual = None

        if not delay_load:
            self.load_model()

    def load_model(self, device=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, '
                  'skipping.'.format(self.vision_tower_name))
            return

        from open_clip.model import CLIP, _build_vision_tower
        config = open_clip.get_model_config(self.vision_tower_name)

        self.visual = _build_vision_tower(
            embed_dim=config['embed_dim'],
            vision_cfg=config['vision_cfg'],
        )
        # print(self.visual.state_dict())

        # pretrained weight
        weights = torch.load(
            f"{self.model_path}/open_clip_pytorch_model.bin",
            )
        # print(weights.keys())

        def get_w(weights, keyword):
            return {k.split(keyword + '.')[1]: v for k, v in weights.items()
                    if keyword in k}

        self.visual.load_state_dict(get_w(weights, 'visual'), strict=False)
        # print(self.visual.state_dict())

        # self.visual = open_clip.create_model(
        #     self.vision_tower_name,
        #     pretrained=f"{self.model_path}/open_clip_pytorch_model.bin",
        # ).visual

        self.visual.trunk.head.global_pool = nn.Identity()
        self.visual.trunk.head.flatten = nn.Identity()
        if device is not None:
            self.visual = self.visual.to(device)
        self.visual.requires_grad_(False)

        self.is_loaded = True

    def extract_cnn_feat(self, image):
        # [bs, 192, 128, 128]
        x = self.visual.trunk.stem(
            image.to(device=self.device, dtype=self.dtype)
        ).contiguous()

        for i in range(4 - 1):  # 倒数第二
            x = self.visual.trunk.stages[i](x)

        vis_loc = x.reshape(*x.shape[:2], -1).permute(0, 2, 1)
        vis_loc = vis_loc.contiguous()
        return vis_loc.to(image.dtype)

    @torch.no_grad()
    def forward(self, image):
        # self.eval()
        return self.extract_cnn_feat(image)

    @property
    def dtype(self):
        return self.visual.trunk.stem[0].weight.dtype

    @property
    def device(self):
        return self.visual.trunk.stem[0].weight.device


class DINOViT(nn.Module):
    # self-supervised： https://github.com/facebookresearch/dinov2
    def __init__(
            self,
            model_name='dinov2_vitg14',
            delay_load=True,
            model_path="playground/checkpoints/openai/facebookresearch_dinov2_main"
    ):
        super().__init__()
        self.is_loaded = False

        self.vision_tower_name = model_name
        self.model_path = model_path
        self.visual = None

        if not delay_load:
            self.load_model()

    def load_model(self, device=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, '
                  'skipping.'.format(self.vision_tower_name))
            return

        self.visual = torch.hub.load(
            self.model_path,
            self.vision_tower_name,
            source='local',
            pretrained=False,
        )
        # pretrained weight
        weights = torch.load(
            f"{self.model_path}/ckpts/dinov2_vitg14_pretrain.pth")
        self.visual.load_state_dict(weights)

        if device is not None:
            self.visual = self.visual.to(device)
        self.visual.requires_grad_(False)

        self.is_loaded = True

    def extract_patch_feat(self, images):
        x = images.to(device=self.device, dtype=self.dtype)

        clip_mean = torch.Tensor([0.48145466, 0.4578275, 0.40821073]).to(
            x, non_blocking=True).view(3, 1, 1)
        clip_std = torch.Tensor([0.26862954, 0.26130258, 0.27577711]).to(
            x, non_blocking=True).view(3, 1, 1)
        dinov2_mean = torch.Tensor([0.485, 0.456, 0.406]).to(
            x, non_blocking=True).view(3, 1, 1)
        dinov2_std = torch.Tensor([0.229, 0.224, 0.225]).to(
            x, non_blocking=True).view(3, 1, 1)

        x = (x * clip_std + clip_mean - dinov2_mean) / dinov2_std
        x = self.visual.prepare_tokens_with_masks(x)

        for blk in self.visual.blocks:  # 最后一层
            x = blk(x)
        # x_norm = self.visual.norm(x)
        x = x[:, 1:]
        return x.to(images.dtype)

    @torch.no_grad()
    def forward(self, images):
        # self.eval()
        return self.extract_patch_feat(images)

    @property
    def dtype(self):
        return self.visual.patch_embed.proj.weight.dtype

    @property
    def device(self):
        return self.visual.patch_embed.proj.weight.device
