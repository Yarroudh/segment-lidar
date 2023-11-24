from setuptools import setup, find_packages
from setuptools.command.install import install
import subprocess

with open("requirements.txt", "r") as file:
    requirements = file.read().splitlines()

class CustomInstallCommand(install):
    def run(self):
        subprocess.call(['pip', 'install', '-r', 'requirements.txt'])
        install.run(self)

        from samgeo import SamGeo
        from samgeo.text_sam import LangSAM

setup(
    name="segment-lidar",
    version='0.2.0',
    description="A package for segmenting LiDAR data using Segment-Anything Model (SAM) from Meta AI Research.",
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    license='BSD 3-Clause "New" or "Revised" License',
    author='Anass Yarroudh',
    author_email='ayarroudh@uliege.be',
    url='https://github.com/Yarroudh/segment-lidar',
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    cmdclass={
        'install': CustomInstallCommand,
    }
)
