#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python train_FRAP.py  --config_path_name './scenarios/config_hz_1.json' --save_dir 'save_model_frap/hz_1'
