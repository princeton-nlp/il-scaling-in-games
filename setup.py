import setuptools

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name="il-scaling-laws",
    version="0.1.0",
    packages=['il_scale'],
    description='Official Implementation of "Scaling Laws for Imitation Learning in NetHack"',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=[
        'wandb',
        'configargparse'
    ],
    python_requires='>=3.7',
    include_package_data=True,
)