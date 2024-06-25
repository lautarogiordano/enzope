"""A setuptools based setup module.

See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
   name="enzope",  # Required

   version="0.0.27",  # Required
   
   description="Agent based modelling in complex networks",
   
   long_description=long_description,
   long_description_content_type="text/markdown",  

   author="Lautaro Giordano",
   author_email="giordanolautaro@gmail.com",
   
   classifiers=[
       "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
   
    keywords="sample, setuptools, development",
   
    package_dir={"": "src"},
    packages=find_packages(where="src"),  # Required

    python_requires=">=3.7, <4",
    install_requires=["numpy >= 1.24",
                      "numba >= 0.0.57",
                      "networkx >= 3.0"],

    project_urls={
        "Source": "https://github.com/lautarogiordano/enzope/",
    },
)