"""Single source of truth for experiment naming."""
import random
import string


def make_experiment_name(clip_model_name, pretrained, dataset, loss, steps, epochs,
                         batch_size, penalty_weight, lr, experiment_name,
                         dynamic_pw=0, dynamic_pw_target_ratio=0.5,
                         dynamic_pw_cosine_decay=False, dynamic_pw_target_ratio_min=None,
                         clean_weight=1.0):
    random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=5))
    duration_str = f'{steps}steps' if steps > 0 else f'{epochs}epochs'
    dpw_str = ''
    if dynamic_pw > 0:
        dpw_str = f'_dpw{dynamic_pw}_t{dynamic_pw_target_ratio}'
        if dynamic_pw_cosine_decay and dynamic_pw_target_ratio_min is not None:
            dpw_str += f'_cd{dynamic_pw_target_ratio_min}'
    cw_str = f'_cw{clean_weight}' if clean_weight != 1.0 else ''
    name = (
        f'{clip_model_name}_{pretrained}_{dataset}_{loss}_'
        f'{duration_str}_bs{batch_size}_pw{penalty_weight}{dpw_str}{cw_str}_'
        f'lr{lr}_{experiment_name}_{random_str}'
    ).replace('/', '_')
    return name


def make_experiment_glob(dataset, loss, epochs, batch_size, penalty_weight, lr,
                         experiment_name, pretrained='openai',
                         clip_model_name='ViT-L-14',
                         dynamic_pw=0, dynamic_pw_target_ratio=0.5,
                         dynamic_pw_cosine_decay=False):
    """Return a glob pattern that matches experiment dirs created by make_experiment_name."""
    dpw_glob = ''
    if int(dynamic_pw) > 0:
        dpw_glob = f'_dpw{dynamic_pw}_t{dynamic_pw_target_ratio}'
        if str(dynamic_pw_cosine_decay).lower() == 'true':
            dpw_glob += '_cd*'
    return (
        f'{clip_model_name}_{pretrained}_{dataset}_{loss}_'
        f'{epochs}epochs_bs{batch_size}_pw{penalty_weight}{dpw_glob}*_'
        f'lr{lr}_{experiment_name}_*'
    ).replace('/', '_')
