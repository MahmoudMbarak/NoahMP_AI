# NoahMP-AI

Code for "A deep learning bias-correction layer for land surface models: Application to soil moisture during drought and hurricane events" https://iopscience.iop.org/article/10.1088/3049-4753/ae7115. A U-Net learns a bias-correction layer that maps Noah-MP land surface model output to SMAP satellite soil moisture.

## Pipeline

```
parallel_simulation_nldas.sh  ──>  ensemble of Noah-MP/HRLDAS runs  ──>  unet.py
   (run_simulations_test.sh = dry-run/sanity check of the same step)
```

## Scripts

| Script | Role |
|--------|------|
| **`parallel_simulation_nldas.sh`** | Generates the training ensemble. Launches many HRLDAS/Noah-MP offline simulations in parallel (one per ensemble member), each with a per-member start date and forcing. Used for the drought case study (Mar–Sep 2022). |
| **`run_simulations_test.sh`** | Test/dry-run version of the runner. Builds and prints the namelist for a few members, asks for confirmation, then runs a small batch — used to validate paths and config before a full ensemble launch. |
| **`unet.py`** | The bias-correction model. Trains a U-Net on the ensemble output (inputs: Noah-MP soil moisture, latent heat, sensible heat) to predict SMAP surface soil moisture. Handles dataset splitting, training, validation metrics (MAE/RMSE/R²), checkpointing, and TensorBoard logging. |

## Configuration

The shell scripts expect a Noah-MP/HRLDAS build (`hrldas.exe`, `namelist.hrldas`,
`NoahmpTable.TBL`) in `WORK_DIR`.
