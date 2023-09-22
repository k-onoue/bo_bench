import warnings
import os

import numpy as np
import pandas as pd
import torch
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.optim import optimize_acqf


def generate_initial_data(target_function, shape=(10,1)):
    train_x = torch.rand(shape).double()
    train_y = target_function(train_x)
    # best_val = train_y.max().item()
    best_val = train_y.min().item()
    return train_x, train_y, best_val


def get_next_points(surrogate, x_train, y_train, best_y, bounds, n_points=1):
    y_train = - y_train

    model = surrogate(x_train, y_train)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)
    
    acq_func = ExpectedImprovement(model=model, best_f=best_y)

    candidates, _ = optimize_acqf(acq_function=acq_func,
                                  bounds=bounds,
                                  q=n_points,
                                  num_restarts=5,
                                  raw_samples=20)
    
    return candidates


def run_bo(n_runs, 
           dim, 
           init_eval_num,
           target_function, 
           surrogate):
    
    res_dict = {}
    init_x, init_y, best_init_y = generate_initial_data(target_function,
                                                        shape=(init_eval_num, dim))
    res_dict['評価点'] = [tuple(x.numpy()) for x in init_x]
    res_dict['評価値'] = [y.numpy()[0] for y in init_y]
    res_dict['最適値'] = [best_init_y if value == best_init_y else np.nan for value in init_y]

    # ここは目的関数の定義域と正解の位置によって変えなければいけない
    bounds = torch.stack([-torch.ones(dim), torch.ones(dim)]).double()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        for i in range(n_runs):
            print(f'Iter: {i+1}')

            new_candidates = get_next_points(surrogate, init_x, init_y, best_init_y, bounds, n_points=1)
            res_dict['評価点'].append(tuple(new_candidates[0].numpy()))

            print(new_candidates)

            new_results = target_function(new_candidates)
            res_dict['評価値'].append(new_results[0][0].numpy())
            
            print(f'New candidates: {new_candidates}')           
            init_x = torch.cat([init_x, new_candidates])
            init_y = torch.cat([init_y, new_results])

            # best_init_y = init_y.max().item()
            best_init_y = init_y.min().item()
            res_dict['最適値'].append(best_init_y)
            print(f'Current best y: {best_init_y}')
            print()

    return pd.DataFrame(res_dict)


def run_expt(settings):
    n_runs = settings['n_runs']
    dim = settings['dim']
    init_eval_num = settings['init_eval_num']
    target_function_name = settings['target_function_name']
    target_function = settings['target_function']
    target_value = settings['target_value']
    surrogate_name = settings['surrogate_name']
    surrogate = settings['surrogate']
    bo_num = settings['bo_num']

    working_dir = '/home/onoue/ws/bo_expts/expts_result/'
    # tmp1 = str(target_function).rfind(".")
    # tmp2 = str(target_function).rfind("(")
    # tmp3 = str(surrogate).rfind(".")
    # tmp4 = str(surrogate).rfind("'")
    # print(str(target_function)[tmp1:tmp2])
    # print(str(surrogate)[tmp3:tmp4])
    # subdir = f'{str(target_function)[tmp1:tmp2]}/{str(surrogate)[tmp3:tmp4]}/{dim}_dim'
    subdir = f'{target_function_name}/{surrogate_name}/{dim}_dim'
    save_dir = os.path.join(working_dir, subdir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    accumulated_loss = 0

    for i in range(bo_num):
        res_df = run_bo(n_runs, dim, init_eval_num, target_function, surrogate)
        accumulated_loss += abs(target_value - res_df['最適値'].min())
        save_path = os.path.join(save_dir, f'opt_{i+1}.csv')
        res_df.to_csv(save_path)

    print(f'総損失: {accumulated_loss}')

    settings['accumulated_loss'] = accumulated_loss
    settings_file_path = os.path.join(save_dir, 'settings.txt') 
    with open(settings_file_path, 'w') as f:
        for key, value in settings.items():
            f.write(f'{key}: {value}\n')



# def run_expt(n_runs, dim, init_eval_num, target_function, surrogate):
    

#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)

#     accumulated_loss = 0

#     # これは target_function によって違うので注意
#     target_value = 0

#     bo_num = 10

#     for i in range(bo_num):
#         res_df = run_bo(n_runs, dim, init_eval_num, target_function, surrogate)
#         accumulated_loss += abs(target_value - res_df['最適値'].min())
#         save_path = os.path.join(save_dir, f'opt_{i+1}.csv')
#         res_df.to_csv(save_path)

#     print(f'総損失: {accumulated_loss}')


#     settings_file_path = os.path.join(save_dir, 'settings.txt') 
#     with open(settings_file_path, 'w') as f:
#         for key, value in settings.items():
#             f.write(f'{key}: {value}\n')