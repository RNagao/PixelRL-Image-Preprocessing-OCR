import os
import logging
import torch
import torch.multiprocessing as mp
import numpy as np
import time
import multiprocessing_logging
from torchinfo import summary
from logging import getLogger
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
from src.models import *
from src.agent import PixelWiseAgent
from src.dataset import create_datasets, create_dataloaders
# from src.train import Trainer
from src.train import train
from tqdm import tqdm

from src.test import test, calculate_reward
from src.share_optim import SharedAdam
from src.state import State
from src.utils import save_model

def main():
    # Setup logger
    multiprocessing_logging.install_mp_handler()
    logger = mp.get_logger()
    # console = logging.StreamHandler()
    # formatter = logging.Formatter('%(name)-16s: %(filename)s %(levelname)-8s %(message)s')
    # console.setFormatter(formatter)
    # logger.addHandler(console)

    # file_handler = logging.FileHandler(f"logger.log", "w")
    # file_handler.setFormatter(logging.Formatter(fmt="%(asctime)s %(levelname)s\t%(message)s"))
    # logger.addHandler(file_handler)

    # logger.setLevel(logging.DEBUG)

    # Hyperparams
    IMG_SIZE = (63, 63)
    BATCH_SIZE = 32
    NUM_WORKERS = 1
    # NUM_WORKERS = 15
    # NUM_WORKERS = 30

    INPUT_SHAPE = 1
    N_ACTIONS = 9
    MOVE_RANGE = 3 # number of actions that move pixel values
    HIDDEN_UNITS = 64
    OUTPUT_SHAPE = INPUT_SHAPE

    EPISODE_SIZE= 3

    MODEl_PATHS = [
         "./torch_initweight/pixelrl.pth",
         "./models/fcn_with_pretrained_pixelrl_30000eps_5steps_0.001lr_0.95gamma/checkpoint_200.pth",
         "./models/fcn_with_pretrained_pixelrl_30000eps_5steps_0.001lr_0.95gamma/checkpoint_250.pth",
         "./models/fcn_30000eps_5steps_0.001lr_0.95gamma/checkpoint_101.pth"
    ]


    # device agnostic code
    # device = "cpu"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create dataloaders
    train_dataset, test_dataset = create_datasets(IMG_SIZE, image_channels=INPUT_SHAPE)
    _, test_dataloader = create_dataloaders(train_dataset=train_dataset,
                                                            test_dataset=test_dataset,
                                                            batch_size=BATCH_SIZE,
                                                            num_workers=0)

    # create model
    fcn = FCN(n_actions=N_ACTIONS,
               num_channels=INPUT_SHAPE,
               hidden_units=HIDDEN_UNITS).to(device)

    # train
    print(f"\nTRAINNING DEVICE: {device}")
    print(f"TEST DATALOADER SIZE: {len(test_dataloader)}\n")

    results = {"default": []}
    results_determiniscally = {"default": []}
    torch.cuda.empty_cache()
    for b, (X, y) in tqdm(enumerate(test_dataloader)):
        state = State(X.shape, MOVE_RANGE, model_hidden_units=HIDDEN_UNITS)
        raw_x = X.cpu().numpy()
        raw_n = np.zeros(raw_x.shape, dtype=np.float32)
        state.reset(raw_x, raw_n)
        results["default"].append(calculate_reward(state.image, y))
        results_determiniscally["default"].append(calculate_reward(state.image, y))
        
    for model_path in MODEl_PATHS:
        model_name = "/".join(model_path.split("/")[-2:])
        if model_name not in results.keys():
                results[model_name] = []
                results_determiniscally[model_name] = []
        fcn.load_state_dict(torch.load(model_path))

        for b, (X, y) in tqdm(enumerate(test_dataloader)):
            results[model_name].append(test(model=fcn,
                                       X=X,
                                       y=y,
                                       episode_size=EPISODE_SIZE,
                                       move_range=MOVE_RANGE,
                                       device=device
                                       ))
            results_determiniscally[model_name].append(test(model=fcn,
                                       X=X,
                                       y=y,
                                       episode_size=EPISODE_SIZE,
                                       move_range=MOVE_RANGE,
                                       device=device
                                       ))

    df = pd.DataFrame(results)
    df_det = pd.DataFrame(results_determiniscally)
    df.to_csv("results.csv")
    df_det.to_csv("results_det.csv")

if __name__ == "__main__":
    main()