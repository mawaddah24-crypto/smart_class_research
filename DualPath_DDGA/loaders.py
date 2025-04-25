# utils/loaders.py

import torch

def load_pretrained_backbone(model, path, verbose=True):
    """
    Memuat bobot EfficientViT pretrained dari VGGFace2 ke dalam backbone model HCAViT.
    """
    state_dict = torch.load(path,map_location='cpu',weights_only=True)

    # Ambil hanya bagian backbone
    backbone_dict = {
        k.replace('backbone.', ''): v for k, v in state_dict.items() if 'backbone.' in k
    }

    missing, unexpected = model.backbone.load_state_dict(backbone_dict, strict=False)

    if verbose:
        print(f"✅ Loaded EfficientViT pretrained on VGGFace2 from: {path}")
        print(f"⏺️ Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
