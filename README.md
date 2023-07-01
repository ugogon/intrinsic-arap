# iARAP

This is the C++ implementation of the paper [[PDF](https://cybertron.cg.tu-berlin.de/projects/iARAP/media/iARAP.pdf)] ["ARAP Revisited: Discretizing the Elastic Energy using Intrinsic Voronoi Cells"](https://cybertron.cg.tu-berlin.de/projects/iARAP/).

Authors: Ugo Finnendahl, Matthias Schwartz, Marc Alexa
Journal: Computer Graphic Forum 2023

## Get the code

Clone the project

```
git clone --recursive https://github.com/ugogon/intrinsic-arap.git
```

## Build the code

Configure with cmake and compile e.g.

```
cd intrinsic-arap
mkdir build
cd build
cmake ..
make -j6
```

# Run the code

Start by e.g.
```
./iARAP_gui ../meshes/bunny.off
```

# Cite
```
@article{10.1111/cgf.14790,
  author = {Finnendahl, Ugo and Schwartz, Matthias and Alexa, Marc},
  title = {ARAP Revisited: Discretizing the Elastic Energy using Intrinsic Voronoi Cells},
  journal = {Computer Graphics Forum},
  volume = {n/a},
  number = {n/a},
  year = {2023},
  month = {4},
  pages = {},
  keywords = {modelling, deformations, polygonal modelling},
  doi = {https://doi.org/10.1111/cgf.14790},
  url = {https://cybertron.cg.tu-berlin.de/projects/iARAP/},
}
```
