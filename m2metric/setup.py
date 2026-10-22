# -*- coding: utf-8 -*-
"""m2metric 打包配置。"""
from setuptools import setup, find_packages

setup(
    name="m2metric",
    version="0.1.0",
    description=("M2 (second spectral moment) as a downstream-product-oriented "
                 "verification metric for data assimilation, exposing the "
                 "blind spot of RMSE for derivative-type diagnostics (Ri/N²/S²/"
                 "vorticity/CAPE)."),
    long_description=open("README.md", encoding="utf8").read(),
    long_description_content_type="text/markdown",
    author="风场—温度—湿度三维精细场实时重构 项目组",
    license="MIT",
    packages=find_packages(),
    install_requires=["numpy>=1.21"],
    python_requires=">=3.8",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
)
