from setuptools import setup, find_packages

setup(
    name="pyarduboy",
    version="0.1.0",
    description="Python library for running Arduboy games using libretro cores",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="puterjam",
    url="https://github.com/puterjam/pyarduboy",
    packages=find_packages(),
    install_requires=[
        "numpy",
    ],
    extras_require={
        "pygame": ["pygame"],
        "luma": ["luma.oled", "luma.core"],
        "audio": ["pyaudio"],
        "evdev": ["evdev"],
    },
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
