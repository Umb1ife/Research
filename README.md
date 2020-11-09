# testFine-Grained Multi-label Image Recognition using Knowledge Graph and Graph Convolutional Networks

## Dependencies

- Python 3.6.10
- PyTorch 1.7.0
- NVIDIA Driver 455.32.00
- CUDA 11.1

## Setup the Environment

- with Docker (Recommended)

  ```bash
  # on the directory contains "Dockerfile"
  docker build . -t myenv:mytag
  docker run -d --gpus all -it -e TZ=Asia/Tokyo --name="mycontainer" --shm-size=32g -v [full-path-of-the-directory-you-mount]:/workspace myenv:mytag

  # enter the container built
  docker exec -it mycontainer bash
  ```

- only pip

  ```bash
  pip install -r requirements.txt
  ```

  After installation with requirement.txt, you need to add two python library, Cartopy and Shapely.

  First, install PROJ and GEOS because Cartopy requires.

  - [PROJ](https://download.osgeo.org/proj/) 6.2.1
  - [GEOS](http://download.osgeo.org/geos/) 3.8.1

  ```bash
  # install Cartopy & Shapely with the option "--no-binary"
  pip install shapely==1.7.1 --no-binary shapely
  pip install cartopy==0.18.0 --no-binary cartopy
  ```
