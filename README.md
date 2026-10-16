# Simulation and example data for "A conditional parameterization framework for turbulent flux estimation across heterogeneous land surfaces"

## Overview

This repository contains the MATLAB implementation of the method proposed in "A conditional parameterization framework for turbulent flux estimation across heterogeneous land surfaces". The data is collected from three stations:
- station 1: Horqin
- station 2: Beijing
- station 3: Huainan

The method can be adjusted by modifying its parameters defined in [API documentation](functions/regressionKernel.m).

## Repository Structure

```text
.
├── README.md
├── main.m
├── method.m
├── data/
│   ├── station_1.mat
│   ├── station_2.mat
│   └── station_3.mat
└── results/
```
