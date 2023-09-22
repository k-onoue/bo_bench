from utils.obj_func import sphere_function
from utils.surrogate import SingleTaskGP_rbf, SingleTaskGP_laplacian
from utils.expt_management import run_expt

from botorch.models import SingleTaskGP

import importlib
import utils.expt_management
importlib.reload(utils.expt_management)

from utils.expt_management import run_expt


dim_1 = 5
n_runs = 100

settings_1 = {
    'n_runs': n_runs,
    'dim': dim_1,
    'init_eval_num': 10,
    'target_function_name': 'sphere_function',
    'target_function': sphere_function,
    'target_value': 0,
    'surrogate_name': 'SingleTaskGP',
    'surrogate': SingleTaskGP,
    'bo_num': 10
}

settings_2 = {
    'n_runs': n_runs,
    'dim': dim_1,
    'init_eval_num': 10,
    'target_function_name': 'sphere_function',
    'target_function': sphere_function,
    'target_value': 0,
    'surrogate_name': 'SingleTaskGP_rbf',
    'surrogate': SingleTaskGP_rbf,
    'bo_num': 10
}

settings_3 = {
    'n_runs': n_runs,
    'dim': dim_1,
    'init_eval_num': 10,
    'target_function_name': 'sphere_function',
    'target_function': sphere_function,
    'target_value': 0,
    'surrogate_name': 'SingleTaskGP_laplacian',
    'surrogate': SingleTaskGP_laplacian,
    'bo_num': 10
}





if __name__ == '__main__':

    # dim_0 = 2
    # settings_1['dim'] = dim_0
    # settings_2['dim'] = dim_0
    # settings_3['dim'] = dim_0
    # run_expt(settings_1)
    # run_expt(settings_2)
    # run_expt(settings_3)


    # dim_1 = 5
    # for i in range(4):
    #     settings_1['dim'] = dim_1
    #     settings_2['dim'] = dim_1
    #     settings_3['dim'] = dim_1
    #     run_expt(settings_1)
    #     run_expt(settings_2)
    #     run_expt(settings_3)
    #     dim_1 += 5


    dim_2 = 25
    for i in range(4):
        settings_1['dim'] = dim_2
        settings_2['dim'] = dim_2
        settings_3['dim'] = dim_2
        run_expt(settings_1)
        run_expt(settings_2)
        run_expt(settings_3)
        dim_2 += 5

    
