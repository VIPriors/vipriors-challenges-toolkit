from setuptools import setup, find_packages


setup(name='vipriors-reid',
      version='0.0.1',
      description='Deep Learning Library for Person Re-identification for the VIPriors Challenge',
      author='Davide Zambrano',
      author_email='davide.zambrano@synergysports.com',
      url='https://github.com/VIPriors/vipriors-challenges-toolkit',
      license='MIT',
      install_requires=[
          'numpy', 'scipy', 'torch==1.8.1', 'torchvision',
          'six', 'h5py', 'Pillow',
          'scikit-learn', 'metric-learn'],
      extras_require={
          'docs': ['sphinx', 'sphinx_rtd_theme'],
      },
      packages=find_packages(),
      keywords=[
          'Person Re-identification',
          'Computer Vision',
          'Deep Learning',
      ])
