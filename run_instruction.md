### detailed instruction
Note:  there are two experiment files in repsitories. Only ONE of them should be put on run, depending on the signal rank in the experiment. experiment.py runs rank 1 experiment, and experiment_rank2.py runs rank 2 experiments.

1. make a branch in [Matrix Completion](https://github.com/adonoho/MatrixCompletion), edit experiment.py as needed, run to make matrix completion data

1. make a new branch of [Matrix Denoising](https://github.com/MiladB90/Matrix_Denoising).  

1. in function test_experiment() change sensing_model_table_name to the name of the table in which you recorded corresponding Matrix_Completion data.  

1. set mc_range for Monte Carlo numbers (mc in data column) you wanna run the experiment.

(the relevant code is copied here)

```
def test_experiment() -> dict:
   
    # below two lines need to modify. it will run denoising model on same grid of hyper-parameters as in sensing_model
    # given.  Monte Carlo numbers would be chosen in mc_range.  since we used mc = 1, ..., 10 for tuning, it's better to
    # use mc >= 11 for making test data.
    sensing_model_table_name = 'milad_mc_0013'
    mc_range = (11, 12)
    ...
```

7. Make necessary changes to experiment.py (or experiment_rank2.py) file, and run.


