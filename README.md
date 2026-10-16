# Simulation and example data for "A conditional parameterization framework for turbulent flux estimation across heterogeneous land surfaces"

## Overview

This repository contains the MATLAB implementation of the method proposed in "A conditional parameterization framework for turbulent flux estimation across heterogeneous land surfaces". An example data is collected from three stations:
- station 1: Horqin
- station 2: Beijing
- station 3: Huainan
Full data set is not available due to privacy.

The data includes: covariate ($X$), factor ($F$), feature ($Z$), and response ($Y$). 

Running [proposed_method.m](proposed_method.m) to estimate the parameter and predict the friction velocity.

The method can be adjusted by modifying its parameters defined in [regressionParaMVC.m](functions/regressionKernel.m).


