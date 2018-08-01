#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(name='gym_textworld',
      version='0.0.1',
      packages=find_packages(),
      scripts=[
          "scripts/tw-make.py",
      ],
      install_requires=open("requirements.txt").readlines())
