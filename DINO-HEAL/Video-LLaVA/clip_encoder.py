import math
from typing import List


import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    CLIPVisionModel,
    CLIPImageProcessor,
    CLIPVisionConfig,
    AutoImageProcessor,
    AutoModel,
)


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        # DINOv2 handles (initialized in load_model)
        self.dinov2_feature_extractor = None
        self.dinov2_model = None

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        # ---- CLIP tower ----
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)
        self.vision_tower.requires_grad_(False)
        self.vision_tower.eval()

        # ---- DINOv2 (cache once) ----
        try:
            # AutoImageProcessor does not accept torch_dtype/low_cpu_mem_usage
            self.dinov2_feature_extractor = AutoImageProcessor.from_pretrained('facebook/dinov2-large')

            # Use fp16 on CUDA to save memory (otherwise fp32 on CPU)
            dino_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            self.dinov2_model = AutoModel.from_pretrained(
                'facebook/dinov2-large',
                torch_dtype=dino_dtype,
                low_cpu_mem_usage=True
            )
            self.dinov2_model.eval()
            self.dinov2_model.to(self.device)
            self.dinov2_model.requires_grad_(False)
        except Exception as e:
            # Graceful degradation: fall back to CLIP-only features if DINOv2 fails to load
            print(f"[CLIPVisionTower] Warning: failed to load DINOv2: {e}")
            self.dinov2_feature_extractor = None
            self.dinov2_model = None

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        """
        Pick hidden states at the requested layer and return either patches-only or cls+patches.
        """
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]  # drop CLS
        elif self.select_feature == 'cls_patch':
            # keep CLS + patches
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        """
        Args:
            images:
                - Tensor: [B,3,H,W] (often CLIP-preprocessed pixel_values)
                - List[Tensor]: each [3,H,W]; will be stacked into a batch

        Returns:
            image_features: Tensor
                - if select_feature == 'patch'    -> [B, N_patch, C]
                - if select_feature == 'cls_patch'-> [B, 1+N_patch, C] (CLS is not fused)
        """
        # Normalize inputs to a batch tensor
        if isinstance(images, list):
            images = torch.stack([img for img in images], dim=0)

        # ---- CLIP forward ----
        clip_out = self.vision_tower(
            images.to(device=self.device, dtype=self.dtype),
            output_hidden_states=True
        )
        image_features = self.feature_select(clip_out).to(images.dtype)

        # If DINOv2 is unavailable, return CLIP features directly
        if (self.dinov2_model is None) or (self.dinov2_feature_extractor is None):
            return image_features

        # ---- DINOv2 saliency via last-layer CLS->patch attention ----

        # 1) Convert (possibly CLIP-normalized) pixel_values back to ~[0,1] for PIL, then to PIL list
        images_for_pil = self._maybe_denormalize_from_clip(images)  # [B,3,H,W], cpu float32 in [0,1]
        images_pil = self._to_pil_list(images_for_pil)

        # 2) DINOv2 preprocessing & forward (request attentions)
        dino_inputs = self.dinov2_feature_extractor(images=images_pil, return_tensors="pt")
        dino_inputs = {k: v.to(self.device, dtype=next(self.dinov2_model.parameters()).dtype)
                       for k, v in dino_inputs.items()}

        dino_outputs = self.dinov2_model(
            **dino_inputs,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True
        )

        # attentions: [num_layers, B, num_heads, S, S] (or similar)
        dino_atts = dino_outputs.attentions
        last_att = dino_atts[-1] if isinstance(dino_atts, (list, tuple)) else dino_atts

        # If shape is [num_layers,B,H,S,S], take the last layer
        if last_att.dim() == 5:
            last_att = last_att[-1]  # [B, H, S, S]

        # Head-averaged attentions
        att = last_att.mean(dim=1)  # [B, S, S]

        # CLS -> patches saliency
        if att.size(1) < 2 or att.size(2) < 2:
            # Defensive fallback
            return image_features
        cls_to_patches = att[:, 0, 1:]  # [B, S-1]

        # 3) Reshape to 2D grid and upsample to CLIP patch grid
        B = images.shape[0]
        H_d, W_d = self._infer_dino_grid(cls_to_patches, dino_inputs)  # DINO grid
        dino_saliency_2d = cls_to_patches.view(B, H_d, W_d)

        # Normalize per-sample to [0,1] for stability
        dino_saliency_2d = (dino_saliency_2d - dino_saliency_2d.amin(dim=(1, 2), keepdim=True)) / \
                           (dino_saliency_2d.amax(dim=(1, 2), keepdim=True) -
                            dino_saliency_2d.amin(dim=(1, 2), keepdim=True) + 1e-6)

        H_c, W_c = self._infer_clip_grid()  # CLIP grid
        dino_saliency_up = F.interpolate(
            dino_saliency_2d.unsqueeze(1),  # [B,1,H_d,W_d]
            size=(H_c, W_c),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)  # [B,H_c,W_c]

        # Flatten to [B, N_clip] and apply sigmoid (kept from original code)
        dino_saliency_flat = dino_saliency_up.view(B, -1)
        dino_saliency_flat = torch.sigmoid(dino_saliency_flat)

        # 4) Align and fuse with CLIP features (keep your original fusion formula)
        if self.select_feature == 'cls_patch':
            # image_features: [B, 1+N, C]
            cls_tok = image_features[:, :1, :]     # [B,1,C]
            patch_feats = image_features[:, 1:, :] # [B,N,C]

            # Global (batch-wide) standardization kept as in the original logic
            patch_feats_norm = (patch_feats - patch_feats.mean()) / (patch_feats.std() + 1e-6)

            saliency_expanded = dino_saliency_flat.unsqueeze(-1).repeat(1, 1, patch_feats.size(-1))  # [B,N,C]
            fused_patch = patch_feats_norm * 0.3 + saliency_expanded * 0.7
            image_features = torch.cat([cls_tok, fused_patch], dim=1)
        else:
            # image_features: [B, N, C]
            image_features_norm = (image_features - image_features.mean()) / (image_features.std() + 1e-6)
            saliency_expanded = dino_saliency_flat.unsqueeze(-1).repeat(1, 1, image_features.size(-1))  # [B,N,C]
            image_features = image_features_norm * 0.3 + saliency_expanded * 0.7

        return image_features

    # ---------------------------
    # Utilities
    # ---------------------------

    def _maybe_denormalize_from_clip(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Attempt to map CLIP-normalized pixel_values back to [0,1] for PIL conversion.
        If values already look like [0,1], just clamp.
        Returns float32 CPU tensor: [B,3,H,W] in [0,1].
        """
        x = pixel_values.detach().to('cpu', dtype=torch.float32)

        # CLIP defaults (openai/clip-vit) means/stds used by many processors
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)

        # Heuristic: if a significant portion is out of [0,1], assume CLIP normalization and denormalize
        ratio_out_of_01 = ((x < 0) | (x > 1)).float().mean().item()
        if ratio_out_of_01 > 0.1:
            x = x * std + mean

        x = x.clamp(0.0, 1.0)
        return x

    def _to_pil_list(self, pixel_values_01: torch.Tensor) -> List:
        """
        Convert [B,3,H,W] in [0,1] to a list of PIL images.
        """
        from torchvision.transforms.functional import to_pil_image
        imgs = []
        for img in pixel_values_01:
            imgs.append(to_pil_image(img))
        return imgs

    def _infer_clip_grid(self):
        """
        Infer CLIP patch grid size (H_c, W_c) from config.
        """
        img_size = self.config.image_size
        patch = self.config.patch_size
        H_c = img_size // patch
        W_c = img_size // patch
        return H_c, W_c

    def _infer_dino_grid(self, cls_to_patches: torch.Tensor, dino_inputs: dict):
        """
        Infer DINO patch grid size using feature_extractor size and model patch_size.
        Fallback to sqrt/factorization if exact grid cannot be derived.
        """
        B, S_minus_1 = cls_to_patches.shape
        H_d = W_d = None
        try:
            # Typical: feature_extractor.size -> 518, patch_size -> 14 => 37x37 tokens
            if hasattr(self.dinov2_feature_extractor, "size"):
                size = self.dinov2_feature_extractor.size
                if isinstance(size, dict) and "shortest_edge" in size:
                    inp = int(size["shortest_edge"])
                elif isinstance(size, int):
                    inp = int(size)
                else:
                    inp = None
            else:
                inp = None

            patch_size = getattr(getattr(self.dinov2_model, "config", None), "patch_size", None)
            if inp is not None and patch_size is not None and patch_size > 0:
                g = inp // patch_size
                if g * g == S_minus_1:
                    H_d = W_d = g
        except Exception:
            pass

        if H_d is None or W_d is None:
            # Fallback: try perfect square, else best factor pair
            g = int(math.sqrt(S_minus_1))
            if g * g == S_minus_1:
                H_d = W_d = g
            else:
                H_d, W_d = self._best_factor_pair(S_minus_1)

        return H_d, W_d

    @staticmethod
    def _best_factor_pair(n: int):
        """
        Find a factor pair (h, w) with minimal difference.
        """
        best = (1, n)
        for i in range(1, int(math.sqrt(n)) + 1):
            if n % i == 0:
                j = n // i
                if abs(i - j) < abs(best[0] - best[1]):
                    best = (i, j)
        return best

    @property
    def dummy_feature(self):
        # Note: if downstream expects [B,N,C], consider returning [1,1,hidden_size] instead.
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
