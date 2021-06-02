[![codecov](https://codecov.io/gh/Welthungerhilfe/cgm-ml/branch/main/graph/badge.svg?token=LG8Q3NTVE1)](https://codecov.io/gh/Welthungerhilfe/cgm-ml)
[![Python package](https://github.com/Welthungerhilfe/cgm-ml/actions/workflows/continous-integration.yml/badge.svg?branch=main)](https://github.com/Welthungerhilfe/cgm-ml/actions/workflows/continous-integration.yml)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Welthungerhilfe/cgm-ml/HEAD)
[![CodeFactor](https://www.codefactor.io/repository/github/welthungerhilfe/cgm-ml/badge)](https://www.codefactor.io/repository/github/welthungerhilfe/cgm-ml)

# Child Growth Monitor Machine Learning

[Child Growth Monitor (CGM)](https://childgrowthmonitor.org) is a
game-changing app to detect malnutrition. If you have questions about the project, reach out to `info@childgrowthmonitor.org`.

This is the Machine Learnine repository associated with the CGM project.

## Introduction

This project uses machine learning to identify malnutrition from 3D scans of children under 5 years of age. This [one-minute video](https://www.youtube.com/watch?v=f2doV43jdwg) explains.

## Getting started

### Requirements

Our development environment is [Microsoft Azure ML](https://azure.microsoft.com/en-us/services/machine-learning/#security)

You will need:
* Python 3
* TensorFlow version 2
* other libraries

To install, run:

```bash
pip install -r requirements.txt
```

For installing point cloud libraries, refer to
[README_installation_details_pcl.md](README_installation_details_pcl.md).

### Dataset access

If you have access to scan data, you can use: `src/data_utils` to understand and visualize the data.

Data access is provided on as-needed basis following signature of the Welthungerhilfe Data Privacy & Commitment to
Maintain Data Secrecy Agreement. If you need data access (e.g. to train your machine learning models),
please contact [Markus Matiaschek](mailto:info@childgrowthmonitor.org) for details.

## Repository structure

The source code is in `src/`.

We make heavy use of AzureML.
For AzureML, all code for an experiment run needs to reside in one directory.
Example: All code for one specific training, e.g. a ResNet training, needs to be in this training directory.

However, many of our trainings (and also evaluation runs) share large portions of code.
In order to reduce code duplication, we copy shared(a.k.a. common) utility code with `copy_dir()` from `src/common/` into the training/evaluation directory.
This way, during the experiment run, the code is in the directory and can be used during the run.

## Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

### Run linting / tests

```bash
# Make sure to be in the root dir of this repository
flake8 src/
pytest
```

## Versioning

Our [releases](https://github.com/Welthungerhilfe/cgm-ml/releases) use [semantic versioning](http://semver.org). You can find a chronologically ordered list of notable changes in [CHANGELOG.md](CHANGELOG.md).

## License

This project is licensed under the GNU General Public License v3.0. See [LICENSE](LICENSE) for details and refer to [NOTICE](NOTICE) for additional licensing notes and use of third-party components.
