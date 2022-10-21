#!/usr/bin/env python
from os import path
from setuptools import find_packages, setup

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

this_directory = path.abspath(path.dirname(__file__)) 


setup(name='pam_ai',
      version='1.0',
      description='Programming and Mathematics for AI - Coursework',
      author='Laerte Adami',
      author_email='laerte.adami@city.ac.uk',
      url='https://github.com/LaerteAdami/pam_ai',
      packages=find_packages(where="src"),
      package_dir={"":"src"}
      #include_package_data=True,
      #install_requires = requirements,
     )
