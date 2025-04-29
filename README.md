# XLModelsXSDatasets

This repository contains code and datasets for training and evaluating predictive models on employment-related data using optimization techniques.

## Contents

- `1_acs_generate_weights.R` / `2_acs_generate_ranks.R` / `3_compare_models.R`: Scripts for generating weights, ranking models, and comparing performance.
- `functions/`: Reusable R functions for estimation, prediction, and data handling.
- `data/`: Education Dataset
- `1.5_complex_opt_generate_weights.jl`: Julia script for complex optimization-based weight generation.

## Setup

1. Install required R packages:
    ```R
    install.packages(c("dplyr", "data.table", "xgboost", "glmnet", "readr"))
    ```

2. (Optional) Julia dependencies:
    ```julia
    using Pkg
    Pkg.add(["JuMP", "Gurobi", "DataFrames", "CSV"])
    ```

3. Set up Git (if cloning):
    ```bash
    git clone https://github.com/Gjain234/XLModelsXSDatasets.git
    ```
## Running

The scripts are intended to be run in the order indicated by their filenames:

1. **Generate weights** using the base script.
2. If you're using the *complex optimization weights* style, run the corresponding script next.  
   (If you're only using the *norms-based* weights, you can skip this step.)
3. Run the **generate ranks** script to compute rankings based on the generated weights.
4. Finally, run the **compare models** script to evaluate performance across approaches.

These scripts can ideally be run in parallel, but this depends on your machine’s capabilities.


## Notes

- This is a sampled version of the full dataset, visit https://github.com/socialfoundations/folktables for full dataset
