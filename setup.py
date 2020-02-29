from setuptools import setup
from setuptools import find_packages
import os


here = os.path.abspath(os.path.dirname(__file__))

try:
    with open(os.path.join(here, "requirements.txt"), encoding="utf-8") as f:
        REQUIRED = f.read().split("\n")
except:
    REQUIRED = []

setup(name='fast_image_classification',
      version='0.1.0',
      description='Tiny library for image classification',
      author='Youness Mansar',
      author_email='mansaryounessecp@gmail.com',
      url='https://github.com/CVxTz/FastImageClassification',
      license='MIT',
      install_requires=REQUIRED,
      classifiers=[
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
          'Topic :: Software Development :: Libraries',
          'Topic :: Software Development :: Libraries :: Python Modules'
      ],
      packages=find_packages(exclude=("example", "app", "data", "docker", "tests")))