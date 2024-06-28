import os
import logging
import torch
import torch.multiprocessing as mp
import numpy as np
import time
import multiprocessing_logging
from torchinfo import summary
from logging import getLogger
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
from src.models import *
from src.agent import PixelWiseAgent
from src.dataset import create_datasets, create_dataloaders
from src.train import Trainer
from src.test import Tester
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
    NUM_WORKERS = 30

    INPUT_SHAPE = 1
    N_ACTIONS = 9
    MOVE_RANGE = 3 # number of actions that move pixel values
    HIDDEN_UNITS = 64
    OUTPUT_SHAPE = INPUT_SHAPE

    LEARNING_RATE = 0.001
    GAMMA = 0.95
    EPISODE_SIZE= 5
    N_EPISODES = 30000

    MODEL_NAME = f"fcn_{N_EPISODES}eps_{EPISODE_SIZE}steps_{LEARNING_RATE}lr_{GAMMA}gamma.pth"
    TARGET_DIR = f"./models/{MODEL_NAME}"

    mp.set_start_method('spawn', force=True)

    # device agnostic code
    device = "cpu"
    # device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create dataloaders
    train_dataset, test_dataset = create_datasets(IMG_SIZE, image_channels=INPUT_SHAPE)
    train_dataloader, test_dataloader = create_dataloaders(train_dataset=train_dataset,
                                                            test_dataset=test_dataset,
                                                            batch_size=BATCH_SIZE,
                                                            num_workers=0)

    # create model
    fcn = FCN(n_actions=N_ACTIONS,
               num_channels=INPUT_SHAPE,
               hidden_units=HIDDEN_UNITS).to(device)
    fcn.load_state_dict(torch.load("./torch_initweight/sig25_gray.pth"))
    
    fcn.share_memory()

    print("\n\nMODEL SUMMARY")
    # summary(model=fcn,
    #     input_size=(1, 1, IMG_SIZE[0], IMG_SIZE[1]),
    #     col_names=["input_size", "output_size", "num_params", "trainable"],
    #     # col_width=20,
    #     row_settings=["var_names"],
    #     device=device)

    # setup optimizer
    optimizer = SharedAdam(params=fcn.parameters(), lr=LEARNING_RATE)
    optimizer.share_memory()

    # setup agent

    # train
    print(f"\nTRAINNING DEVICE: {device}")
    print(f"NUM_WORKERS: {NUM_WORKERS}")
    print(f"TRAIN DATALOADER SIZE: {len(train_dataloader)}")
    print(f"TEST DATALOADER SIZE: {len(test_dataloader)}\n")

    fcn.train()

    global_avg_train_rewards = mp.Array('d', len(train_dataloader))
    global_avg_test_rewards = mp.Array('d', len(train_dataloader))

    train_start = time.time()
    fcn, ep = load_checkpoint(TARGET_DIR, fcn)

    # 0: local | 1: koyeb | 2: lambda
    processes_running = mp.Array('i',3)

    while ep < N_EPISODES:
        process_not_completed = [i for i in range(len(train_dataloader))]
        torch.cuda.empty_cache()
        while len(process_not_completed) > 0:
            workers = []
            for b, (X, y) in enumerate(train_dataloader):
                if b in process_not_completed:
                    workers.append(Trainer(
                                    process_idx=b,
                                    model=fcn,
                                    optimizer=optimizer,
                                    X=X,
                                    y=y,
                                    n_episodes=ep,
                                    episode_size=EPISODE_SIZE,
                                    lr=LEARNING_RATE,
                                    gamma=GAMMA,
                                    move_range=MOVE_RANGE,
                                    img_size=IMG_SIZE,
                                    batch_size=BATCH_SIZE,
                                    device=device,
                                    logger=logger,
                                    model_hidden_units=HIDDEN_UNITS,
                                    global_avg_train_rewards=global_avg_train_rewards,
                                    running_processes=processes_running
                            ))
                    
                    if len(workers) >= NUM_WORKERS or len(workers) == len(process_not_completed):
                        [w.start() for w in workers]
                        [w.join() for w in workers]
                        success_processes = [w.process_idx for w in workers if w.exitcode == 0]
                        print(f"SUCCESS: {success_processes}")
                        workers = []
                        torch.cuda.empty_cache()
                        if success_processes:
                            process_not_completed = [p for p in process_not_completed if p not in success_processes]
                            success_processes = []
        save_model(model=fcn,
               target_dir=TARGET_DIR,
               model_name=f"checkpoint_{ep}.pth")
        print(f"SAVED CHECKPOINT {ep}")
        ep += 1
    train_stop = time.time()

    workers=[]
    process_not_completed = [i for i in range(len(train_dataloader))]
    torch.cuda.empty_cache()
    while process_not_completed:
        for b, (X, y) in enumerate(test_dataloader):
            if b in process_not_completed:
                if len(workers) >= NUM_WORKERS or len(workers) == len(process_not_completed):
                        [w.start() for w in workers]
                        [w.join() for w in workers]
                        success_processes = [(i, w) for i, w in enumerate(workers) if w.exitcode == 0]
                        workers = []
                        torch.cuda.empty_cache()
                        if success_processes:
                            process_not_completed = [p for p in process_not_completed if p not in success_processes]

                workers.append(Tester(
                                process_idx=b,
                                model=fcn,
                                optimizer=optimizer,
                                X=X,
                                y=y,
                                episode_size=EPISODE_SIZE,
                                lr=LEARNING_RATE,
                                gamma=GAMMA,
                                move_range=MOVE_RANGE,
                                img_size=IMG_SIZE,
                                batch_size=BATCH_SIZE,
                                device=device,
                                logger=logger,
                                global_avg_test_rewards=global_avg_test_rewards
                            ))
    test_stop = time.time()

    print(f"\nTRAIN BEST REWARD: {np.mean(global_avg_train_rewards)}\n\nTEST BEST REWARD: {np.mean(global_avg_test_rewards)}")
    print(f"\nTRAIN TIME: {train_stop - train_start}")
    print(f"\nTEST TIME: {test_stop - train_stop}")
    print(f"\nTOTAL TIME: {test_stop - train_start}")

    save_model(model=fcn,
               target_dir=TARGET_DIR,
               model_name=MODEL_NAME)
    

def load_checkpoint(target_dir, model):
    checkpoints_paths = sorted(list(Path(target_dir).rglob('checkpoint*')))
    if len(checkpoints_paths) == 0:
        return model, 0

    last_checkpoint = None
    i = 0
    for checkpoint in checkpoints_paths: 
        checkpoint_i = int(checkpoint.name.split('_')[-1].split('.')[0]) + 1
        if checkpoint_i > i:
            i = checkpoint_i
            last_checkpoint = checkpoint

    model.load_state_dict(torch.load(last_checkpoint))

    return model, i


if __name__ == "__main__":
    main()