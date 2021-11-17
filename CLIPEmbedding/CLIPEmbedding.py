from transformers import CLIPProcessor, CLIPModel
import torch
from torch import Tensor
import numpy as np
from PIL.Image import Image

from typing import Union, Iterable, Optional
from pathlib import Path


class CLIPEmbedding:
    def __init__(
        self, 
        model_path: Union[str, Path] = "openai/clip-vit-base-patch32",
        device: Union[str, torch.device] = torch.device('cpu')
        ):
        """
        Arguments:
        ----------
        model_path: 🤗 Transformers URI or Path to checkpoint directory.
        device: 'cpu' or 'cuda'
        
        See https://huggingface.co/transformers/model_doc/clip.html
        """
        if type(device) == str:
            device = torch.device(device)

        self.device = device
        self.model = CLIPModel.from_pretrained(model_path).to(device)
        self.model = self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.processor = CLIPProcessor.from_pretrained(model_path)


    def embed_images(self, images: Iterable[Union[Image, np.array, Tensor]]):
        """
        Arguments:
        ----------
        images: List of images (PIL, Numpy, or Pytorch)
        
        Returns:
        --------
        outputs: torch.Tensor of shape (n_images, 512)
        """
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


    def embed_text(self, text: Iterable[str]):
        """
        Arguments:
        ----------
        text: List of strings
        
        Returns:
        --------
        outputs: torch.Tensor of shape (n_text, 512)
        """
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
        """
        Arguments:
        ----------
        images: torch.Tensor of shape (n_images, n_dims)
        text: torch.Tensor of shape (n_text, n_dims)
        
        Returns:
        --------
        probability: torch.Tensor of shape (n_images, n_text). Rows sum to 1.
        """
        images = images / images.norm(dim=-1, keepdim=True)
        text = text / text.norm(dim=-1, keepdim=True)

        logit_scale = self.model.logit_scale.exp()
        logits = logit_scale * images @ text.t()
        probability = torch.softmax(logits, dim=-1)
        
        return probability
      
