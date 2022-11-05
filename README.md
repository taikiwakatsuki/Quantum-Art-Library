# Quantum-Art-Library

[![Python Versions](https://img.shields.io/pypi/pyversions/quantumartlibrary.svg)](https://pypi.org/project/quantumartlibrary/)
[![PyPI version](https://badge.fury.io/py/quantumartlibrary.svg)](https://badge.fury.io/py/quantumartlibrary)
[![PyPI - License](https://img.shields.io/pypi/l/quantumartlibrary.svg)](https://pypi.org/project/quantumartlibrary/)

This is a creative coding library for artistic expression of quantum behavior.

{: align="center"}
![DoubleSlit](https://user-images.githubusercontent.com/52993310/200101709-5bd06789-2e93-45c2-ab7e-a7dfccf1c52b.gif)

## Requirement
* NumPy
* Matplotlib
* PyOpenGL
* glfw
* cv2
* PIL
* tqdm
* os
* platform
* CuPy (optional)

## Installation
```
pip install QuantumArtLibrary
```

## Usage
There are two ways to use it. One is to output the quantum simulation results as an array. The other is the output of images and videos.

### Quantum Simulation
```
simulation(image="sample.png", n=400, step=300, p=[0, -15], v=[0, 80], method="split-step", white=1, range=[0.1, 0.9])
```

Not all arguments are required. The above is the default value other than image. If no image is specified, it simulates a double slit experiment.

**image**: *Image to input.*<br>
**`n`**: *Number of grid points.*<br>
**`step`**: *Number of frames to output.*<br>
**`p`**: *Initial position of the quantum.*<br>
**`v`**: *Quantum speed and direction.*<br>
**`method`**: *Simulation method. When simulating on a GPU using CuPy, you can use the following methods:* `split-step-cupy`<br>
**`white`**: *Set 1 for images based on white and 0 for images based on black.*<br>
**`range`**: *Set the range to convert to grayscale.*


### Output of images and videos
```
drawing(simulation, vertex_shader="shader.vert", fragment_shader="shader.frag")
```

Take the output of the simulation function as an argument. Shaders are not required, but can be set.

**`vertet_shader`**: *Set the vertex shader. Refer to `shader.vert`*<br>
**`fragment_shader`**: *Set fragment shader. Refer to `shader.frag`*
