# 3D Computer Vision (SoSe2023)
Some code examples used or demonstrated in the 3D Computer Vision course in summer term 2023

## Prerequisites

First we need to download and install Python 3 from [python.org](https://www.python.org/downloads/). The recommended version is `3.10.7` or above. To inspect the currently installed version we can use the following terminal command:

```shell
python --version
```

*Ensure that the correct version is used in the terminal as well as the selected interpreter.*

---

Usually the dependency management tool `pip` is included in the Python installation. If it is not, download and install it from [pip.pypa.io](https://pip.pypa.io/en/stable/installation/). Version `19.3` or above is recommended. We can again use a terminal command to check the version:

```shell
pip --version
```

Create an virtual environment in order to avoid installing the necessary modules globally. In the project root folder execute the following terminal command:

```shell
python -m venv .venv
```

The will create the hidden folder `.venv` in your current project folder. Next we need to enable the VENV. The VENV will only be active for the current terminal session. So closing the terminal and re-opening it, will disable the VENV.

**Always make sure to enable it before you start developing.**

```shell
# For Windows use
.venv\Scripts\Activate.ps1

# For Linux
source .venv/bin/activate
```

Special care is needed when using Windows. Please consult the following [guide][venv-guide] for more information.

---

On both Operation Systems the VENV can be deactivated with the following terminal command:

```shell
deactivate
```

### Installing dependencies

Installing the required dependencies is straight forward:

```shell
pip install .
```

This will load the defined dependencies in `pyproject.toml` and install them inside the VENV.

## Content

[3DCV_Qt_Stereo.py](./3DCV_Qt_Stereo.py) is a tool to illustrate the epipolar geometry and shows how to compute the fundamental matrix. On start the fundamental matrix is calculated from features in two images. You can click points in one image and the corresponding epipolar line is drawn in the other.

The images can also be rectified.

In the lower part, the features including their matches can be shown as well as all epipolar lines for all found feature points.