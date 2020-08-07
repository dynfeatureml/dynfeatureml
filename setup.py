# Copyright (C) 2020 Prashant Nair <prashant.nair2050@gmail.com>
# License: MIT, prashant.nair2050@gmail.com

from setuptools import setup

def readme():
    with open('README.md') as f:
        README = f.read()
    return README

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="dynfeatureml",
    version="1.0",
    description="Dynamic Feature Machine Learning - A direct way to handle dynamic number of features to train your machine learning model",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/dynfeatureml/dynfeatureml",
    author="Prashant Nair",
    author_email="prashant.nair2050@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    packages=["dynfeatureml"],
    include_package_data=True,
    install_requires=required
)