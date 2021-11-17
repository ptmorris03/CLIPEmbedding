# CLIPEmbedding
Easy text-image embedding and similarity with pretrained CLIP in PyTorch

### Example

```python3
from CLIPEmbedding import CLIPEmbedding
from skimage.io import imread

clip = CLIPEmbedding(
    model_path="openai/clip-vit-base-patch16", #🤗 Transformers URI or Path to checkpoint file
    device='cuda' #defaults to 'cpu'
)

im = imread("https://picsum.photos/224/224")

im = clip.embed_images([im])
text = clip.embed_text(["red", "blue", "green", "yellow"])

scores = clip.similarity(im, text)
```

### Setup
  - Install preferred version of pytorch (`'cpu'` or `'cuda'`) before installing `requirements.txt`
