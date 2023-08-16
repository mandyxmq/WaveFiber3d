
## Overview

This repository includes the simulation and fitting implementation of the SIGGRAPH 2023 paper **A Practical Wave Optics Reflection Model for Hair and Fur**, by Mengqi (Mandy) Xia, Bruce Walter, Christophe Hery, Olivier Maury, Eric Michielssen, and Steve Marschner.

We developed a 3D wave optics simulator based on a physical optics approximation, using a GPU-based hierarchical algorithm to greatly accelerate the calculation. To practically handle geometry variations in the scene, we propose a model based on wavelet noise, capturing the important statistical features in the simulation results that are relevant for rendering. Our compact noise model can be easily combined with existing scattering models to render hair and fur of various colors, introducing visually important colorful glints that were missing from all previous models.

More information about this project can be found at the [project website](https://mandyxmq.github.io/research/wavefiber_3d.html).


## Simulation

compareBEM2d directory contains the validation against 2D BEM

compile rough2d.cu: 
 ```
    nvcc rough2d.cu -rdc=true -o rough2d
 ```
run rough2d: 
 ```
./rough2d [testcase] [polarization] [lambdastart] [lambdaend]
 ``` 
where testcase=0 is circle, testcase=1 is ellipse; polarization: M is TM polarization, E is TE polarization; 
lambdastart and lambdaend are the indices for the starting wavelength index and the ending index (included). They are in the range of [0,24].
rough circle:  radius 10um, roughness 0.02, 1000 elements; ior 1.55+0.1j is used in the BEM simulation
rough ellipse:  radii 16um, 10um, roughness 0.05, 2000 elements; ior 1.55+0.1j is used in the BEM simulation

2D BEM:
We compare with 2D BEM simulation using the code published by Xia et al. 2020 (https://github.com/mandyxmq/WaveOpticsFiber). 

---

compareMie directory contains the validation against Mie scattering

compile sphere.cu: 
 ```
    nvcc sphere.cu -rdc=true -o sphere -lcufft
 ```
run sphere: 
 ```
    ./sphere [lambdastart] [lambdaend] [depth] 
 ```
where lambdastart and lambdaend are the indices for the starting wavelength index and the ending index (included). They are in the range of [0, 24];
depth is the maximal depth of the tree. When depth is 0, the code computes wave scattering without the tree structure.

Mie scattering:
We compare with Mie scattering using the python library PyMieScatt. We include the script pymiescatt.ipynb in the same directory.

---

compareBEM3d contains the validation against 3D BEM

compile ellipsoid.cu: 
 ```
    nvcc ellipsoid.cu -rdc=true -o ellipsoid -lcufft
 ```
Running ellipsoid is similiar to the sphere example described above.

3D BEM:
We compare with 3D BEM simulation using an open source library called Bempp (https://bempp.com/), by Betcke et al. 
We include the script ellipsoid_bempp.py in the same directory.

---

fibersim directory contains treePO.cu code and mesh of fiber segments to compute scattering patterns from these fiber segments.
compare treePO.cu: 
 ```
    nvcc treePO.cu -rdc=true -o treePO -lcufft
 ```
run treePO: 
 ```
    ./treePO [lambdastart] [lambdaend] [thetastart] [phistart] [depth] [startindex] [endindex]
 ```
where lambdastart and lambdaend are the indices for the starting wavelength index and the ending index (included). They are in the range of [0,24].
thetastart is the start index of incident theta angle. We use 180 incident theta angles. 
phistart is the start index of incident phi angle. We use 360 incident phi angles.
depth is the maximal depth of the tree. When depth is 0, the code computes wave scattering without the tree structure.
startindex and endindex can be used to restrict the geometry construction to partial fiber (not the full circle).

Please download the [fiber geometry](https://drive.google.com/drive/folders/10vlg475xcVyQCvK9M51ZczmhJpTvpbf6?usp=sharing) and put them under fibersim/mesh/.

---

Note that you can run other 2D or 3D objects using our PO simulator as well. 
For 2D objects, you need to provide the vertex positions of the line segments that descritized the 2D boundary.
For 3D objects, you need to input the coordinates of the points, as well as the normal and the area attached to each point.

---

## Fitting

In fitting directory, we provide the code to fit wavelet noise to the granular patterns in fiber.
waveletnoise_single.cpp generates noise instances for each frequency band.
fitnoise.ipynb compute the weights for the frequency bands, calling functions in fittingutil.py.

## Contact

If you have any questions regarding this work, please contact Mandy Xia (mengqi.xia AT epfl.ch).








