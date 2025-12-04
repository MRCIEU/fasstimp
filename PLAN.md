# Plan: Implement `fasstimp` Package for GWAS Imputation

We will formalize the existing R scripts into a robust R package, ensuring dependencies are managed, functions are documented, and validation workflows are established using the provided example data and simulation tools.

## Steps

1. **Package Infrastructure**: Update `DESCRIPTION` to include missing dependencies (`glue`, `dplyr`, `tibble`, `ggplot2`, `glmnet`) and configure `NAMESPACE` for exports.
2. **Code Refinement**: Refactor `R/fasstimp.r` and `R/sim.r` to add `roxygen2` documentation, ensuring `perform_imputation` and `simulate_ss` are properly exported and documented.
3. **Validation Workflow (Real Data)**: Create a vignette/script that loads `inst/extdata/ebi-a-GCST005812`, masks a subset of known SNPs, runs `perform_imputation`, and calculates correlation/RMSE between imputed and actual values.
4. **Validation Workflow (Simulation)**: Create a second workflow using `R/sim.r` to generate synthetic summary stats from LD reference data, mask SNPs, and validate imputation accuracy in a controlled setting.
5. **Unit Testing**: Implement `testthat` unit tests for core functions to ensure stability, checking edge cases like missing columns or mismatched LD matrices.

## Further Considerations

1. **Performance Metrics**: Sandardize the output metrics (e.g., $r^2$, RMSE) for the validation steps.
2. **Data Loading**: The example data paths in `inst/extdata` will need helper functions to be easily loaded by end-users. 
