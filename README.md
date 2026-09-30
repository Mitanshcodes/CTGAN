<div align="center">
<br/>
<p align="center">
    <i>This repository is part of <a href="https://sdv.dev">The Synthetic Data Vault Project</a>, a project from <a href="https://datacebo.com">DataCebo</a>.</i>
</p>

[![Development Status](https://img.shields.io/badge/Development%20Status-2%20--%20Pre--Alpha-yellow)](https://pypi.org/search/?c=Development+Status+%3A%3A+2+-+Pre-Alpha)
[![PyPI Shield](https://img.shields.io/pypi/v/ctgan.svg)](https://pypi.python.org/pypi/ctgan)
[![Unit Tests](https://github.com/sdv-dev/CTGAN/actions/workflows/unit.yml/badge.svg)](https://github.com/sdv-dev/CTGAN/actions/workflows/unit.yml)
[![Downloads](https://pepy.tech/badge/ctgan)](https://pepy.tech/project/ctgan)
[![Coverage Status](https://codecov.io/gh/sdv-dev/CTGAN/branch/main/graph/badge.svg)](https://codecov.io/gh/sdv-dev/CTGAN)
[![Forum](https://img.shields.io/badge/Forum-Join%20now!-36C5F0)](https://forum.datacebo.com)

<div align="left">
<br/>
<p align="center">
<a href="https://github.com/sdv-dev/CTGAN">
<img align="center" width=40% src="https://github.com/sdv-dev/SDV/blob/stable/docs/images/CTGAN-DataCebo.png"></img>
</a>
</p>
</div>

</div>

# Overview

CTGAN is a collection of Deep Learning based synthetic data generators for single table data, which are able to learn from real data and generate synthetic data with high fidelity.

| Important Links                               |                                                                                           |
| --------------------------------------------- | ------------------------------------------------------------------------------------------|
| :computer: **[Website]**                      | Check out the SDV Website for more information about our overall synthetic data ecosystem.|
| :orange_book: **[Blog]**                      | A deeper look at open source, synthetic data creation and evaluation.                     |
| :book: **[Documentation]**                    | Quickstarts, User and Development Guides, and API Reference.                              |
| :octocat: **[Repository]**                    | The link to the Github Repository of this library.                                        |
| :keyboard: **[Development Status]**           | This software is in its Pre-Alpha stage.                                                  |
| :busts_in_silhouette: **[DataCebo Forum]**    | Discuss CTGAN features, ask questions, and receive help.                                  |

[Website]: https://sdv.dev
[Blog]: https://datacebo.com/blog
[Documentation]: https://bit.ly/sdv-docs
[Repository]: https://github.com/sdv-dev/CTGAN
[License]: https://github.com/sdv-dev/CTGAN/blob/main/LICENSE
[Development Status]: https://pypi.org/search/?c=Development+Status+%3A%3A+2+-+Pre-Alpha
[DataCebo Forum]: https://forum.datacebo.com

Currently, this library implements the **CTGAN** and **TVAE** models described in the [Modeling Tabular data using Conditional GAN](https://arxiv.org/abs/1907.00503) paper, presented at the 2019 NeurIPS conference.

# Install

## Use CTGAN through the SDV library

:warning: If you're just getting started with synthetic data, we recommend installing the SDV library which provides user-friendly APIs for accessing CTGAN. :warning:

The SDV library provides wrappers for preprocessing your data as well as additional usability features like constraints. See the [SDV documentation](https://bit.ly/sdv-docs) to get started.

## Use the CTGAN standalone library

Alternatively, you can also install and use **CTGAN** directly, as a standalone library:

**Using `pip`:**

```bash
pip install ctgan
```

**Using `conda`:**

```bash
conda install -c pytorch -c conda-forge ctgan
```

When using the CTGAN library directly, you may need to manually preprocess your data into the correct format, for example:

* Continuous data must be represented as floats
* Discrete data must be represented as ints or strings
* The data should not contain any missing values

# Usage Example

In this example we load the [Adult Census Dataset](https://archive.ics.uci.edu/ml/datasets/adult)* which is a built-in demo dataset. We use CTGAN to learn from the real data and then generate some synthetic data.

```python3
from ctgan import CTGAN
from ctgan import load_demo

real_data = load_demo()

# Names of the columns that are discrete
discrete_columns = [
    'workclass',
    'education',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'native-country',
    'income',
]

ctgan = CTGAN(epochs=10)
ctgan.fit(real_data, discrete_columns)

# Create synthetic data
synthetic_data = ctgan.sample(1000)
```

*For more information about the dataset see:
Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml].
Irvine, CA: University of California, School of Information and Computer Science.

# Command-line usage

From a source checkout, create an environment and install the package dependencies first:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

If creating the environment reports that `ensurepip` is unavailable, and your system already
has pip, use this alternative:

```bash
python3 -m venv --without-pip .venv
python3 -m pip --python .venv/bin/python install pip
source .venv/bin/activate
python -m pip install -e .
```

Activate the environment again with `source .venv/bin/activate` in each new terminal.
The examples below use `input.csv` as a placeholder; replace it with an existing dataset path.

Train on CSV data and generate 1,000 synthetic rows:

```bash
python -m ctgan input.csv synthetic.csv --infer-discrete --epochs 300 --num-samples 1000 --save model.pkl
```

`--infer-discrete` treats text and boolean columns as categorical. Numeric columns remain
continuous. For numeric category codes, use an explicit list such as `--discrete city,status_code`
instead, including every categorical column. Inference cannot be combined with `--discrete`,
`--metadata`, or `--tsv`. Dates, identifiers, and free text still require deliberate preprocessing;
inference does not provide a universal schema for these types.

CTGAN already encodes categories as one-hot vectors and normalizes continuous values internally.
There is no need to replace categories with arbitrary numbers before training.

Reuse a saved model without retraining:

```bash
python -m ctgan input.csv synthetic.csv --load model.pkl --num-samples 1000
```

The input path is still required and read; its row count supplies the default sample count when
`--num-samples` is omitted. Training settings apply only when creating a new model.
`--no-header` reads and writes headerless CSV files; explicit discrete columns then use zero-based
indices. Sample counts, epochs, embedding dimensions, and layer sizes must be positive integers,
and batch sizes must be divisible by 10.

TSV mode requires `--tsv --metadata schema.txt`. The schema has one line per column, for example
`C 0 100` for a continuous column and `D {red blue}` for a categorical column. Output uses tabs and
restores the category labels. CSV metadata instead uses JSON with a `columns` list containing
`name` and `type` entries; types other than `continuous` are treated as categorical.

Conditional sampling requires both `--sample_condition_column` and
`--sample_condition_column_value`. It increases the likelihood of the requested category without
guaranteeing it for every row.

# Join our community

Join our [forum](https://forum.datacebo.com/) to discuss more about CTGAN, ask questions, and receive help.

**Interested in contributing to CTGAN?** Read our [Contribution Guide](CONTRIBUTING.rst) to get started.

# Citing CTGAN

If you use CTGAN, please cite the following work:

*Lei Xu, Maria Skoularidou, Alfredo Cuesta-Infante, Kalyan Veeramachaneni.* **Modeling Tabular data using Conditional GAN**. NeurIPS, 2019.

```LaTeX
@inproceedings{ctgan,
  title={Modeling Tabular data using Conditional GAN},
  author={Xu, Lei and Skoularidou, Maria and Cuesta-Infante, Alfredo and Veeramachaneni, Kalyan},
  booktitle={Advances in Neural Information Processing Systems},
  year={2019}
}
```

# Related Projects
Please note that these projects are external to the SDV Ecosystem. They are not affiliated with or maintained by DataCebo.

* **R Interface for CTGAN**: A wrapper around **CTGAN** that brings the functionalities to **R** users.
More details can be found in the corresponding repository: https://github.com/kasaai/ctgan
* **CTGAN Server CLI**: A package to easily deploy CTGAN onto a remote server. Created by Timothy Pillow @oregonpillow at: https://github.com/oregonpillow/ctgan-server-cli

---


<div align="center">
<a href="https://datacebo.com"><img align="center" width=40% src="https://github.com/sdv-dev/SDV/blob/stable/docs/images/DataCebo.png"></img></a>
</div>
<br/>
<br/>

[The Synthetic Data Vault Project](https://sdv.dev) was first created at MIT's [Data to AI Lab](
https://dai.lids.mit.edu/) in 2016. After 4 years of research and traction with enterprise, we
created [DataCebo](https://datacebo.com) in 2020 with the goal of growing the project.
Today, DataCebo is the proud developer of SDV, the largest ecosystem for
synthetic data generation & evaluation. It is home to multiple libraries that support synthetic
data, including:

* 🔄 Data discovery & transformation. Reverse the transforms to reproduce realistic data.
* 🧠 Multiple machine learning models -- ranging from Copulas to Deep Learning -- to create tabular,
  multi table and time series data.
* 📊 Measuring quality and privacy of synthetic data, and comparing different synthetic data
  generation models.

[Get started using the SDV package](https://sdv.dev/SDV/getting_started/install.html) -- a fully
integrated solution and your one-stop shop for synthetic data. Or, use the standalone libraries
for specific needs.
