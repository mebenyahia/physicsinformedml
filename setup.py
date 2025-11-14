"""Setup script for physicsinformedml package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="physicsinformedml",
    version="1.0.0",
    author="mebenyahia",
    description="A comprehensive demonstration of Physics-Informed Neural Networks (PINNs)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mebenyahia/physicsinformedml",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    keywords="physics-informed neural-networks machine-learning pde deep-learning pytorch",
    project_urls={
        "Bug Reports": "https://github.com/mebenyahia/physicsinformedml/issues",
        "Source": "https://github.com/mebenyahia/physicsinformedml",
    },
)
