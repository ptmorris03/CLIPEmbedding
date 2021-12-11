# CLIPEmbedding
Easy text-image embedding and similarity with pretrained CLIP in PyTorch

### Example

```python3
from CLIPEmbedding import CLIPEmbedding
from skimage.io import imread

clip = CLIPEmbedding(
    #model_path="openai/clip-vit-base-patch32", #🤗 Transformers URI or Path to checkpoint file
    #device='cuda' #defaults to 'cpu'
)

im = imread("https://picsum.photos/224/224")

im = clip.embed_images([im]) #list of PIL Images, numpy arrays, or torch Tensors
text = clip.embed_text(["red", "blue", "green", "yellow"])

print(im.shape) #(1, 512)
print(text.shape) #(4, 512)

scores = clip.similarity(im, text, softmax=True)
print(scores.shape) #(1, 4)
```

### Setup
  - Install preferred version of pytorch (`'cpu'` or `'cuda'`) before installing
  - `pip install git+https://github.com/pmorris2012/CLIPEmbedding.git`
