import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms


class ClipVisionModel(nn.Module):
    def __init__(self, model, normalize):
        super().__init__()
        self.model = model
        self.normalize = normalize

    def forward(self, vision, output_normalize=False):
        embedding = self.model(self.normalize(vision))
        if output_normalize:
            embedding = F.normalize(embedding, dim=-1)
        return embedding


class MLCDVisionModel(nn.Module):
    def __init__(self, model, normalize):
        super().__init__()
        self.model = model
        self.normalize = normalize

    def forward(self, vision, output_normalize=False):
        embedding = self.model(self.normalize(vision)).pooler_output
        if output_normalize:
            embedding = F.normalize(embedding, dim=-1)
        return embedding


def load_clip_orig(clip_model_name):
    """Load the original CLIP model and split its image processor into preprocess + normalize.
    Also returns the appropriate tokenizer."""
    if clip_model_name == "DFN":
        model, image_processor = open_clip.create_model_from_pretrained(
            'hf-hub:apple/DFN2B-CLIP-ViT-L-14', device='cpu')
        tokenizer = open_clip.tokenize
    elif clip_model_name == "SigLip":
        model, image_processor = open_clip.create_model_from_pretrained(
            'hf-hub:timm/ViT-SO400M-14-SigLIP', device='cpu')
        tokenizer = open_clip.get_tokenizer('hf-hub:timm/ViT-SO400M-14-SigLIP')
    elif clip_model_name == "MetaCLIP":
        model, image_processor = open_clip.create_model_from_pretrained(
            'ViT-L-14-quickgelu', pretrained='metaclip_400m', device='cpu')
        tokenizer = open_clip.tokenize
    else:
        model, image_processor = open_clip.create_model_from_pretrained(
            clip_model_name, pretrained='openai')
        tokenizer = open_clip.tokenize

    preprocess = transforms.Compose(image_processor.transforms[:-1])
    normalize = image_processor.transforms[-1]
    del image_processor
    return model, preprocess, normalize, tokenizer


def load_vision_model(vision_model):
    """Load the frozen vision model (DINOv2 or MLCD)."""
    if vision_model == "dino":
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
    elif vision_model == "mlcd":
        from transformers import AutoModel
        model = AutoModel.from_pretrained('DeepGlint-AI/mlcd-vit-large-patch14-336')
    else:
        raise ValueError(f'Unknown vision model: {vision_model}')
    model.eval()
    return model


def build_dino_normalize(clip_model_name, vision_model, clip_normalize):
    """Build the normalization transform for the vision model, matching training logic."""
    if clip_model_name == "ViT-L-14-336" and vision_model == "dino":
        return torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=224,
                                          interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
                                          max_size=None, antialias=True),
            clip_normalize])
    elif clip_model_name == "ViT-L-14-336" and vision_model == "mlcd":
        return clip_normalize
    elif vision_model == "dino":
        return clip_normalize
    else:
        return torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=336,
                                          interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
                                          max_size=None, antialias=True),
            clip_normalize])


def wrap_vision_model(raw_model, clip_model_name, vision_model, clip_normalize):
    """Wrap a raw vision model with the correct normalization, returning a ClipVisionModel or MLCDVisionModel."""
    normalize = build_dino_normalize(clip_model_name, vision_model, clip_normalize)
    if vision_model == "dino":
        return ClipVisionModel(model=raw_model, normalize=normalize)
    else:
        return MLCDVisionModel(model=raw_model, normalize=normalize)
