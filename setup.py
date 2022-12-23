#!/usr/bin/env python
# coding: utf-8

from setuptools import setup
from genius_chatbot.version import __version__, __author__
from pathlib import Path
import re


readme = Path('README.md').read_text()
version = __version__
readme = re.sub(r"Version: [0-9]*\.[0-9]*\.[0-9][0-9]*", f"Version: {version}", readme)
print(f"README: {readme}")
with open("README.md", "w") as readme_file:
    readme_file.write(readme)
description = 'Use huggingface models to create an intelligent and scalable chatbot'

setup(
    name='genius-chatbot',
    version=f"{version}",
    description=description,
    long_description=f'{readme}',
    long_description_content_type='text/markdown',
    url='https://github.com/Knuckles-Team/genius-chatbot',
    author=__author__,
    author_email='knucklessg1@gmail.com',
    license='Unlicense',
    packages=['genius_chatbot'],
    include_package_data=True,
    install_requires=['torch>=1.13.1', 'transformers>=4.25.1', 'accelerate>=0.15.0', 'psutil>=5.9.4'],
    py_modules=['genius_chatbot'],
    package_data={'genius_chatbot': ['genius_chatbot']},
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'License :: Public Domain',
        'Environment :: Console',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    entry_points={'console_scripts': ['genius-chatbot = genius_chatbot.genius_chatbot:main']},
)
