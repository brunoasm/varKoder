from setuptools import setup, find_packages

setup(
    name="varKoder",
    version="0.9.3",
    packages=find_packages(),
    url="https://github.com/brunoasm/varKoder",
    license="GNU General Public License",
    author="Bruno de Medeiros", 
    author_email="bdemedeiros@fieldmuseum.org", 
    description="A tool that uses variation in K-mer frequencies as DNA barcodes.",
    install_requires=[
        "torch",
        "fastai>=2.7.13",
        "timm",
        "pyarrow",
        "pandas",
        "humanfriendly",
        "tenacity",
    ],
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "varKoder=varKoder.varKoder:main"  
        ]
    },
)
