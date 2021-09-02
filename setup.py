from distutils.core import setup

setup(name='bitinfer',
      author='Piero Kauffmann',
      description='This package provides an easy way to deploy a large pool of'
                  ' custom fine-tuned BERT models with BitFit ([Zaken et. al, '
                  'arXiv:2106.10199](https://arxiv.org/abs/2106.10199)) in a '
                  'single Pytorch model instance without any speed penalty.',
      url='https://github.com/piero2c/bitinfer',
      version='0.1',
      packages=['bitinfer'])
