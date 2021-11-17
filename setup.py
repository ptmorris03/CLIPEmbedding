from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name='CLIPEmbedding',
    version='0.0.1',
    description='Easy text-image embedding and similarity with pretrained CLIP in PyTorch ',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/pmorris2012/CLIPEmbedding',
    author='Paul Morris',
    author_email='pmorris2012@fau.edu',
    packages=find_packages(where='CLIPEmbedding'),
    python_requires='>=3.6, <4',
    install_requires=['ftfy', 'numpy', 'transformers']
)
