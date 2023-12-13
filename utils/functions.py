import os
import torch
import numpy as np
import pandas as pd
import random
import logging
import copy
from .metrics import Metrics

def set_torch_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def set_output_path(args):
    # args.output_path가 존재하지 않는 경우, 해당 경로를 생성합니다.
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # 예측 결과를 저장할 경로를 args.output_path와 args.logger_name을 결합하여 생성합니다.
    pred_output_path = os.path.join(args.output_path, args.logger_name)
    # 해당 경로가 존재하지 않는 경우, 경로를 생성합니다.
    if not os.path.exists(pred_output_path):
        os.makedirs(pred_output_path)

    # 모델을 저장할 경로를 pred_output_path와 args.model_path를 결합하여 생성합니다.
    model_path = os.path.join(pred_output_path, args.model_path)
    # 해당 경로가 존재하지 않는 경우, 경로를 생성합니다.
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # 생성된 예측 결과 경로와 모델 저장 경로를 반환합니다.
    return pred_output_path, model_path
