from transformers import CLIPProcessor, CLIPModel
import torch
from torch import Tensor
import numpy as np
from PIL import Image

from typing import Union, Iterable, Optional
from pathlib import Path


class CLIPEmbedding:
    def __init__(
        self, 
        model_path: Union[str, Path] = "openai/clip-vit-base-patch32",
        device: Union[str, torch.device] = torch.device('cpu')
        ):
        if type(device) == str:
            device = torch.device(device)

        self.device = device
        self.model = CLIPModel.from_pretrained(model_path).to(device)
        self.model = self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.processor = CLIPProcessor.from_pretrained(model_path)


    def embed_images(
        self,
        images: Iterable[Union[Image.Image, np.array, Tensor]]
        ):
        inputs = self.processor(
            images=images,
            return_tensors="pt",
            padding=True
        )

        for key in inputs:
            inputs[key] = inputs[key].to(self.device)

        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)

        return outputs


    def embed_text(
        self,
        text: Iterable[Union[Image.Image, np.array, Tensor]]
        ):
        inputs = self.processor(
            text=text,
            return_tensors="pt",
            padding=True
        )

        for key in inputs:
            inputs[key] = inputs[key].to(self.device)

        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)

        return outputs


    def similarity(self, images: Tensor, text: Tensor):
        images = images / images.norm(dim=-1, keepdim=True)
        text = text / text.norm(dim=-1, keepdim=True)

        logit_scale = self.model.logit_scale.exp()
        logits = logit_scale * images @ text.t()
        return torch.softmax(logits, dim=-1)
      
