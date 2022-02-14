# Local Linearity Regularizer for Training Verifiably Robust Models

This repository contains an implementation of the Local Linearity Regularizer
(LLR) using TensorFlow v1:
[https://arxiv.org/abs/1907.02610](https://arxiv.org/abs/1907.02610).

## Installation

1. Ensure you have a Python 3.7 environment. (The required version of TensorFlow
   is not supported from 3.8.) Tools such as
   [pyenv](https://github.com/pyenv/pyenv#basic-github-checkout) can facilitate
   multiple Python versions:

   1. Clone the pyenv repository:

      ```bash
      git clone https://github.com/pyenv/pyenv.git ~/.pyenv
      ```

   1. Configure shell environment for pyenv, as described at
      https://github.com/pyenv/pyenv#basic-github-checkout

   1. Install Python 3.7 into pyenv:

      ```bash
      pyenv install 3.7.7
      ```

   1. Set Python 3.7 as the current version:

      ```bash
      pyenv shell 3.7.7
      ```
      (To revert to the system version, use `pyenv shell default`.)

1. Clone the repository:

   ```bash
   git clone https://github.com/deepmind/local_linearity_regularizer.git
   cd local_linearity_regularizer
   ```

1. Recommended: set up a virtual Python environment:

   ```bash
   python3 -m venv llr_env
   source llr_env/bin/activate
   ```
   (To leave the virtual environment, type `deactivate`.)

1. Install the dependencies:

   ```bash
   pip3 install -r requirements.txt
   ```

## Usage

The following command trains a model on CIFAR-10 with epsilon set to 8/255:

```bash
python3 -m local_linearity_regularizer.train_main --config=local_linearity_regularizer/config.py
```

## Citing this work

If you use this code in your work, we ask that you cite our paper with the
following bibtex:

```
@misc{qin2019adversarial,
    title={Adversarial Robustness through Local Linearization},
    author={Chongli Qin and James Martens and Sven Gowal and Dilip Krishnan
        and Krishnamurthy Dvijotham and Alhussein Fawzi and Soham De and
        Robert Stanforth and Pushmeet Kohli},
    year={2019},
    eprint={1907.02610},
    archivePrefix={arXiv},
    primaryClass={stat.ML}
}
```

## Disclaimer

This is not an official Google product.
