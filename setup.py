#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='CLIPEmbedding',
      version='0.0.1',
      description='Easy text-image embedding and similarity with pretrained CLIP in PyTorch.',
      author='Paul Morris',
      author_email='pmorris2012@fau.edu',
      packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
      license='LICENSE.txt',
      install_requires=['ftfy', 'numpy', 'transformers']
    )
