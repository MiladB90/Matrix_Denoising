import numpy as np

def shrinker(x: float, shrinker_name: str, shrinker_parameters_list: list) -> float:
    if shrinker_name == 'soft_thresholding':
        # parameters[0] = soft thresholding level
        level = shrinker_parameters_list[0]
        return max(0, x - level)



def get_shrinker_name_and_parameters(p: float, solver_name: str, solver_parameters_list: list) -> list:
    if solver_name == 'norm_nuc_pen':
        shrinker_name = 'soft_thresholding'

        lambda_mc = solver_parameters_list[0]
        shrinker_parameters = [2 * ((1 / np.sqrt(p)) - 1) + lambda_mc]


    return shrinker_name, shrinker_parameters
