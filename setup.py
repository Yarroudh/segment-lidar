from setuptools import setup, find_packages

with open("requirements.txt", "r") as file:
    requirements = file.read().splitlines()

setup(
    name="segment-lidar",
    version='0.1.2',
    description="A package for segmenting LiDAR data using Segment-Anything Model (SAM) from Meta AI Research.",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    license='BSD 3-Clause "New" or "Revised" License',
    author = 'Anass Yarroudh',
    author_email = 'ayarroudh@uliege.be',
    url = 'https://github.com/Yarroudh/segment-lidar',
    packages=find_packages(),
    install_requires=requirements,
    dependency_links=[
        'git+https://github.com/jianboqi/CSF.git'
    ],
    entry_points={
        "console_scripts": [
            "segment-lidar=segment_lidar.main:cli"
        ]
    }
)
