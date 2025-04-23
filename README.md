# CryptoNet MNIST Classification

A privacy-preserving implementation of CryptoNet for MNIST digit classification using homomorphic encryption.

## Overview

This project implements a CryptoNet model that can perform classification on encrypted MNIST images using the CKKS homomorphic encryption scheme through TenSEAL.

## Features

- Homomorphic encryption of MNIST dataset using CKKS scheme
- Efficient data chunking for large datasets
- CryptoNet architecture implementation
- Privacy-preserving training pipeline
- Multi-core processing support

## Requirements

- Python 3.7+
- PyTorch
- TenSEAL
- torchvision
- numpy
- tqdm

## Installation

```bash
pip install torch torchvision tenseal numpy tqdm
```

## Usage

1. Encrypt the MNIST dataset:
```bash
python enc.py
```

2. Train the CryptoNet model:
```bash
python cryptonet.py
```

## Project Structure

- `enc.py`: Script for encrypting MNIST dataset
- `cryptonet.py`: CryptoNet implementation and training
- `README.md`: Project documentation

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MNIST dataset
- TenSEAL library
- PyTorch 