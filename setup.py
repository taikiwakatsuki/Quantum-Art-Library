import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="QuantumArtLibrary",
    author="taikiwakatsuki",
    author_email="mutual_bookers.0q@icloud.com",
    description="Creative coding library for artistic expression of quantum behavior.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/taikiwakatsuki/Quantum-Art-Library",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy",
        "matplotlib",
        "tqdm",
        "PyOpenGL",
        "glfw",
        "opencv-python",
        "Pillow",
    ]
)
