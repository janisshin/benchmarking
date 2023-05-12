# benchmarking

Protocol:

1. Run `generated_synthetic_models.ipynb`
2. Run `cobra2tellurium.py`
3. Run `filtering_synthetic_models.ipynb`
4. Run `generated_model_data.ipynb`
5. Edit `run_inference_pipeline.py` and then run in the terminal 

Data Omission Codes
| Code |          Data Type Omitted         |
|:----:|:----------------------------------:|
|   A  |                None                |
|   B  |               fluxes               |
|   C  |            enzyme levels           |
|   D  | internal metabolite concentrations |
|   E  | external metabolite concentrations |


