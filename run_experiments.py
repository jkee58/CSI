# Obtained from: https://github.com/lhoyer/DAFormer
# Modifications:
# - Add startup test
# - Save configs as python file instead of json file
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

import argparse
import logging
import os
import subprocess
import uuid
from datetime import datetime

import torch
from experiments import generate_experiment_cfgs
from mmengine.config import Config
from mmengine.utils import get_git_hash
from tools import train
import copy


def run_command(command):
    p = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
    for line in iter(p.stdout.readline, b''):
        print(line.decode('utf-8'), end='')


def rsync(src, dst):
    rsync_cmd = f'rsync -a {src} {dst}'
    print(rsync_cmd)
    run_command(rsync_cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--exp',
        type=int,
        default=None,
        help='Experiment id as defined in experiment.py',
    )
    group.add_argument(
        '--config',
        default=None,
        help='Path to config file',
    )
    parser.add_argument(
        '--ngpus', type=int, default=None, help='Number of GPUs')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--startup-test', action='store_true')
    args = parser.parse_args()
    assert (args.config
            is None) != (args.exp
                         is None), 'Either config or exp has to be defined.'

    GEN_CONFIG_DIR = 'configs/generated/'
    JOB_DIR = 'jobs'
    cfgs, config_files = [], []

    # Training with Predefined Config
    if args.config is not None:
        cfg = Config.fromfile(args.config)
        # Specify Name and Work Directory
        exp_name = f'local-{cfg["exp"]}'
        unique_name = f'{datetime.now().strftime("%y%m%d_%H%M")}_' \
                      f'{cfg["name"]}_{str(uuid.uuid4())[:5]}'
        child_cfg = {
            '_base_': args.config.replace('configs', '../..'),
            'name': unique_name,
            'work_dir': os.path.join('work_dirs', exp_name, unique_name),
            'git_rev': get_git_hash()
        }
        cfg_out_file = f"{GEN_CONFIG_DIR}/{exp_name}/{child_cfg['name']}.py"
        os.makedirs(os.path.dirname(cfg_out_file), exist_ok=True)
        assert not os.path.isfile(cfg_out_file)
        with open(cfg_out_file, 'w') as f:
            f.write(Config(child_cfg).pretty_text)
        # Json files do not distinguish tuples and lists, which is
        # necessary for Resize.img_scale.
        # with open(cfg_out_file, 'w') as of:
        #     json.dump(child_cfg, of, indent=4)
        config_files.append(cfg_out_file)
        cfgs.append(cfg)

    # Generate configs from experiments.py
    if args.exp is not None:
        exp_name = f'local-exp{args.exp}'
        if args.startup_test:
            exp_name += '-startup'
        cfgs = generate_experiment_cfgs(args.exp)

        for i, cfg in enumerate(cfgs):
            assert isinstance(cfg, dict)
            if args.debug:
                cfg.setdefault('default_hooks', {})['logger'] = dict(
                    type='LoggerHook', interval=10)
                cfg['evaluation'] = dict(interval=200, metric='mIoU')
                # cfg['train_cfg'] = dict(type='IterBasedTrainLoop', max_iters=iters, val_interval=200)
                if 'dacs' in cfg['name'] or 'minent' in cfg['name'] or \
                        'advseg' in cfg['name']:
                    cfg.setdefault('uda', {})['debug_img_interval'] = 10
                    # cfg.setdefault('uda', {})['print_grad_magnitude'] = True

            # Support a startup test
            if args.startup_test:
                cfg['log_level'] = logging.INFO
                cfg['train_cfg'] = dict(
                    type='IterBasedTrainLoop', max_iters=20, val_interval=10)
                cfg['default_hooks'] = dict(
                    logger=dict(type='LoggerHook', interval=10))
                cfg['custom_hooks'] = [
                    dict(
                        type='CustomDebugHook',
                        priority='ABOVE_NORMAL',
                        interval=10),
                    # dict(type='WandbCommitHook',
                    #      priority='BELOW_NORMAL',
                    #      interval=1)
                ]
            # Override number of GPUs
            if args.ngpus:
                cfg['number_of_gpus'] = args.ngpus
                cfg['model_wrapper_cfg'] = dict(
                    type='UDAModelWrapper', static_graph=True)

            # Generate a meta information
            cfg['name'] = f'{datetime.now().strftime("%y%m%d_%H%M")}_' \
                          f'{cfg["name"]}_{str(uuid.uuid4())[:5]}'
            cfg['work_dir'] = os.path.join('work_dirs', exp_name, cfg['name'])
            cfg['git_rev'] = get_git_hash()
            cfg['_base_'] = ['../../' + e for e in cfg['_base_']]

            # Save config to file
            cfg_out_file = f"{GEN_CONFIG_DIR}/{exp_name}/{cfg['name']}.py"
            os.makedirs(os.path.dirname(cfg_out_file), exist_ok=True)
            assert not os.path.isfile(cfg_out_file)
            with open(cfg_out_file, 'w') as f:
                f.write(Config(cfg).pretty_text)
            # Json files do not distinguish tuples and lists, which is
            # necessary for Resize.img_scale.
            # with open(cfg_out_file, 'w') as of:
            #     json.dump(cfg, of, indent=4)
            config_files.append(cfg_out_file)

    # Train from config files
    for i, cfg in enumerate(cfgs):
        if args.startup_test and cfg['randomness']['seed'] != 0:
            continue

        number_of_gpus = cfg.get('number_of_gpus', 1)
        if number_of_gpus > 1:
            # Multi-GPU training
            os.environ["OMP_NUM_THREADS"] = '16'
            command = f'bash tools/dist_train.sh {config_files[i]} {number_of_gpus}'
            print(f'Run {command}')
            run_command(command)
        elif number_of_gpus == 1:
            # Single-GPU training
            print(f'Run train.main {cfg["name"]}')
            print(config_files[i])
            train.main([config_files[i]])
        else:
            raise NotImplementedError(f'{args.ngpus} GPUs not supported')

        torch.cuda.empty_cache()
