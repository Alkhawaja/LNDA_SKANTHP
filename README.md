# LNDA_SKANTHP

This repository contains implementations of different topic models and temporal processes:

Topic models
- **LDA**: Latent Dirichlet Allocation
- **LNDA**: Latent Nested Dirichlet Allocation
Temporal processes
- **HP**: Hawkes Process
- **STHP**: Structured Transformer Hawkes Process
- **SKANTHP**: Structured KAN Transformer Hawkes Process

## File Structure

```
src/
│
├── Constants.py      # Hyperparameters and global constants
├── Dataset.py        # Dataset loading and preprocessing
├── Layers.py         # Transformer or neural network layers
├── LNDA.py           # Implementation of LNDA model
├── Models.py         # Common model interfaces
├── Modules.py        # Model submodules
├── Preprocess.py     # Data preprocessing logic
├── SKANTHP.py        # SKANTHP model logic
├── SubLayers.py      # Attention and feed-forward components
```

### Requirements
Install dependencies:

```bash
pip install -r requirements.txt
```

### Running an Experiment

```bash
python main.py -data REDD -num_types 3 -model SKANTHP
```

### Running all Experiments for all datasets with 3 iterations

```bash
run.ps1
```

## Optional Arguments

| Argument                     | Description                                   | Default  |
|------------------------------|-----------------------------------------------|----------|
| `-data`                      | Dataset to use                                | `'REDD'` |
| `-num_types (energy levels)` | Number of appliance types to model            | `3`      |
| `-iteration (optional)`      | Trial index or random seed, can be left empty | ``       |

## Output

The results (metrics and visualizations) will be saved in the `{dataset}/` directory and figs will be saved to `figs/` subdirectory.
