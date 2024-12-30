import numpy as np

def shrinker(x: float, shrinker_name: str, shrinker_parameters_list: list) -> float:
    if shrinker_name == 'no_shrink':
        return x
    elif shrinker_name == 'soft_thresholding':
        # parameters[0] = soft thresholding level
        level = shrinker_parameters_list[0]
        return max(0, x - level)



def get_shrinker_name_and_parameters(p: float, n:int, sigma: float,
                                     solver_name: str, solver_parameters_list: list, tune_mode: str) -> list:
    if tune_mode == 'no_shrink':
        shrinker_name = 'no_shrink'
        shrinker_parameters = []
    elif solver_name == 'norm_nuc_pen':
        shrinker_name = 'soft_thresholding'

        # tune modes
        if tune_mode == 'theory':
            lambda_mc = solver_parameters_list[0]
            # both below way should give the same, but for some reason they don't. check later
            # pen_to_milad_coef = 1 / (sigma * np.sqrt(n))
            # lambda_mdn = round(2 * ((1 / np.sqrt(p)) - 1) + pen_to_milad_coef * lambda_mc, 3)
            lambda_mdn = round(2 * (1 / np.sqrt(p) - 1) + 5 * np.sqrt(p), 3)
            shrinker_parameters = [lambda_mdn]
        if tune_mode == 'empirical':
            shrinker_parameters = [11.80]


    return shrinker_name, shrinker_parameters
