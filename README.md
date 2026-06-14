# FD Kit


A collection of Python functions and tools for exploring ecological and trait-based data. 
Includes utilities for calculating functional diversity metrics (FRic, FEve, FDiv, FDis, RaoQ) 
and performing data preprocessing for further analysis.

## Papers to Read

- Botta-Dukát, Z. (2005), Rao's quadratic entropy as a measure of functional diversity based on multiple traits. Journal of Vegetation Science, 16: 533-540. https://doi.org/10.1111/j.1654-1103.2005.tb02393.x
- Laliberté, E. and Legendre, P. (2010), A distance-based framework for measuring functional diversity from multiple traits. Ecology, 91: 299-305. https://doi.org/10.1890/08-2244.1 
- Mouchet, M.A., Villéger, S., Mason, N.W.H. and Mouillot, D. (2010), Functional diversity measures: an overview of their redundancy and their ability to discriminate community assembly rules. Functional Ecology, 24: 867-876. https://doi.org/10.1111/j.1365-2435.2010.01695.x 
- Villéger, S., Mason, N.W.H. and Mouillot, D. (2008), NEW MULTIDIMENSIONAL FUNCTIONAL DIVERSITY INDICES FOR A MULTIFACETED FRAMEWORK IN FUNCTIONAL ECOLOGY. Ecology, 89: 2290-2301. https://doi.org/10.1890/07-1206.1 

## Things to do: 
- Write a test script / Notebook 
- Documentation
- Clean up utils file. Remove what is not necessary 
- Add docstrings to each function 
- Add additional arguments for each argument

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Functions](#functions)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:

```bash
git clone  https://github.com/qiulinlx/FD_Playground.git 

```

2. Install dependencies (recommended to use a virtual environment):

```bash

pip install -r requirements.txt

```

### Usage

```python
import numpy as np
from fdiv import functional_richness, functional_evenness

traits = np.array([[1,2],[2,3],[3,4]])
abundances = [0.5, 0.3, 0.2]

FRic = functional_richness(traits)
FEve = functional_evenness(traits, abundances)

print("FRic:", FRic)
print("FEve:", FEve)
```

## Examples

Check out `examples/functional_diversity_demo.ipynb` for step-by-step usage of the functions.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

MIT License

