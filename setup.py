from setuptools import setup, find_packages

setup(
    name="varKoder",
    version="0.13.0",
    packages=find_packages(),
    url="https://github.com/brunoasm/varKoder",
    license="GNU General Public License",
    author="Bruno de Medeiros", 
    author_email="bdemedeiros@fieldmuseum.org", 
    description="A tool that uses variation in K-mer frequencies as DNA barcodes.",
    install_requires=[
        "torch",
        "fastai",
        "timm",
        "pyarrow",
        "pandas",
        "humanfriendly",
        "tenacity",
        "huggingface_hub[fastai]"
    ],
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "varKoder=varKoder.varKoder:main"  
        ]
    },
)
