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
You will want to run in the order the files are marked in. Generate weights first, then the complex opt generate weights if you are using that weights style. If you just want the norms weights, no need to run this. Then the generate ranks script, and finally the compare models script. These files should ideally be run in parallel, but depending on your machine that may or may not be possible.

## Notes

- This is a sampled version of the full dataset, visit https://github.com/socialfoundations/folktables for full dataset
