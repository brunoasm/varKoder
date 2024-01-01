from setuptools import setup, find_packages

setup(
    name='varKoder',
    version='0.9.0',  # Update based on your versioning scheme
    packages=find_packages(),
    url='https://github.com/brunoasm/varKoder',
    license='GNU General Public License',  # Specify GPL version if applicable
    author='Bruno de Medeiros',  # Replace with your name
    author_email='bdemedeiros@fieldmuseum.org',  # Optionally, replace with your email
    description='A tool that uses variation in K-mer frequencies as DNA barcodes.',
    install_requires=[
        'torch',
        'fastai>=2.7.12',
        'timm==0.6.13',
        'pyarrow',
        'pandas',
        'humanfriendly',
        'tenacity'
    ],
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'varKoder=varKoder.varKoder:main'  # Adjust if using a different function name
        ]
    }
)

