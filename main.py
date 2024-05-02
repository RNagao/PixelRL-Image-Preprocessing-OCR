import os
import logging
import torch
import torch.multiprocessing as mp
import multiprocessing_logging
from torchinfo import summary
from logging import getLogger

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
from src.models import FCN
from src.agent import PixelWiseAgent
from src.dataset import create_datasets, create_dataloaders
from src.train import Trainer
from src.test import Tester
from src.share_optim import ShareAdam
from src.state import State

def main():
    # Setup logger
    multiprocessing_logging.install_mp_handler()
    logger = mp.get_logger()
    console = logging.StreamHandler()
    formatter = logging.Formatter('%(name)-16s: %(filename)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logger.addHandler(console)

    file_handler = logging.FileHandler(f"logger.log", "w")
    file_handler.setFormatter(logging.Formatter(fmt="%(asctime)s %(levelname)s\t%(message)s"))
    logger.addHandler(file_handler)

    logger.setLevel(logging.DEBUG)

    # Hyperparams
    IMG_SIZE = (481, 321)
    BATCH_SIZE = 16
    # NUM_WORKERS = os.cpu_count()
    NUM_WORKERS = 1

    INPUT_SHAPE = 1
    N_ACTIONS = 14
    MOVE_RANGE = 3 # number of actions that move pixel values
    HIDDEN_UNITS = 64
    OUTPUT_SHAPE = INPUT_SHAPE

    LEARNING_RATE = 0.001
    GAMMA = 0.95
    EPISODE_SIZE=3
    N_EPISODES =2

    mp.set_start_method('spawn', force=True)

    # device agnostic code
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"

    # Create dataloaders
    train_dataset, test_dataset = create_datasets(IMG_SIZE, image_channels=INPUT_SHAPE)
    train_dataloader, test_dataloader = create_dataloaders(train_dataset=train_dataset,
                                                            test_dataset=test_dataset,
                                                            batch_size=BATCH_SIZE,
                                                            num_workers=0)

    # create model
    fcn = FCN(n_actions=N_ACTIONS,
                input_shape=INPUT_SHAPE,
                hidden_units=HIDDEN_UNITS,
                output_shape=OUTPUT_SHAPE).to(device)
    fcn.share_memory()

    # print("\n\nMODEL SUMMARY")
    # summary(model=fcn,
    #     input_size=(1, 1, IMG_SIZE[0], IMG_SIZE[1]),
    #     col_names=["input_size", "output_size", "num_params", "trainable"],
    #     col_width=20,
    #     row_settings=["var_names"],
    #     device=device)

    # setup optimizer
    optimizer = ShareAdam(params=fcn.parameters(), lr=LEARNING_RATE)
    # optimizer = torch.optim.Adam(params=fcn.parameters(), lr=LEARNING_RATE)

    # setup agent
    agent = PixelWiseAgent(model=fcn,
                            optimizer=optimizer,
                            lr=LEARNING_RATE,
                            t_max=EPISODE_SIZE,
                            gamma=GAMMA,
                            batch_size=BATCH_SIZE,
                            img_size=IMG_SIZE,
                            device=device,
                            logger=logger)

    # train
    print(f"\nTRAINNING DEVICE: {device}")
    print(f"NUM_WORKERS: {NUM_WORKERS}")
    print(f"TRAIN DATALOADER SIZE: {len(train_dataloader)}")
    print(f"TEST DATALOADER SIZE: {len(test_dataloader)}\n")

    torch.cuda.empty_cache()
    fcn.train()

    workers = []
    for b, (X, y) in enumerate(train_dataloader):
        workers.append(Trainer(
                        process_idx=b,
                        agent=agent,
                        # optimizer=optimizer,
                        X=X,
                        y=y,
                        n_episodes=N_EPISODES,
                        episode_size=EPISODE_SIZE,
                        lr=LEARNING_RATE,
                        gamma=GAMMA,
                        move_range=MOVE_RANGE,
                        img_size=IMG_SIZE,
                        batch_size=BATCH_SIZE,
                        device=device,
                        logger=logger
                    ))
        if len(workers) >= NUM_WORKERS:
            [w.start() for w in workers]
            [w.join() for w in workers]
            killed_processes = [(i, w) for i, w in enumerate(workers) if w.exitcode != 0]
            workers = []
            torch.cuda.empty_cache()
            if killed_processes:
                workers = [Trainer(
                        process_idx=b,
                        agent=agent,
                        # optimizer=optimizer,
                        X=w.X,
                        y=w.y,
                        n_episodes=N_EPISODES,
                        episode_size=EPISODE_SIZE,
                        lr=LEARNING_RATE,
                        gamma=GAMMA,
                        move_range=MOVE_RANGE,
                        img_size=IMG_SIZE,
                        batch_size=BATCH_SIZE,
                        device=device,
                        logger=logger
                    )
                    for b, w in killed_processes]

    while len(workers) > 0:
        [w.start() for w in workers]
        [w.join() for w in workers]
        torch.cuda.empty_cache()
        killed_processes = [(i, w) for i, w in enumerate(workers) if w.exitcode != 0]
        workers = []
        if killed_processes:
                workers = [Trainer(
                        process_idx=b,
                        agent=agent,
                        # optimizer=optimizer,
                        X=w.X,
                        y=w.y,
                        n_episodes=N_EPISODES,
                        episode_size=EPISODE_SIZE,
                        lr=LEARNING_RATE,
                        gamma=GAMMA,
                        move_range=MOVE_RANGE,
                        img_size=IMG_SIZE,
                        batch_size=BATCH_SIZE,
                        device=device,
                        logger=logger
                    )
                    for b, w in killed_processes]

    # p_count=0
    # workers=[]
    # for b, (X, y) in enumerate(test_dataloader):
    #     workers.append(Tester(
    #                     process_idx=b,
    #                     agent=agent,
    #                     X=X,
    #                     y=y,
    #                     episode_size=EPISODE_SIZE,
    #                     gamma=GAMMA,
    #                     move_range=MOVE_RANGE,
    #                     img_size=IMG_SIZE,
    #                     batch_size=BATCH_SIZE,
    #                     device=device,
    #                     logger=logger
    #                 ))
    #     if p_count < NUM_WORKERS:
    #         p_count += 1
    #     else:
    #         [w.start() for w in workers]
    #         [w.join() for w in workers]
    #         p_count = 0
    #         workers = []

    #         torch.cuda.empty_cache()

    # if len(test_dataloader) % NUM_WORKERS != 0:
    #     [w.start() for w in workers]
    #     [w.join() for w in workers]
    #     p_count = 0
    #     workers = []
    #     torch.cuda.empty_cache()


    print(f"\nTRAIN BEST REWARD: {fcn.train_average_max_reward}\n\nTEST BEST REWARD: {fcn.test_average_max_reward}")

if __name__ == "__main__":
    main()