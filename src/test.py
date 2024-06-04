import torch
import numpy as np
import logging
import torch.multiprocessing as mp
from tqdm.auto import tqdm
from typing import Tuple
from statistics import fmean

from src.agent import PixelWiseAgent
from src.state import State
from src.reader import compare_strings_levenshtein, read_image_array_words

class Tester(mp.Process):
    def __init__(self,
                process_idx: int,
                model: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                X,
                y,
                gamma:float,
                episode_size: int,
                lr:float,
                move_range:int,
                logger:logging.Logger,
                global_avg_test_rewards,
                img_size:tuple[int,int]=(481, 321),
                batch_size:int=32,
                device:torch.device="cpu"):
        super().__init__()

        self.process_idx = process_idx
        self.agent = PixelWiseAgent(model=model,
                            optimizer=optimizer,
                            lr=lr,
                            t_max=episode_size,
                            gamma=gamma,
                            batch_size=batch_size,
                            img_size=img_size,
                            device=device,
                            logger=logger)
        self.X = X
        self.y = y
        self.gamma = gamma
        self.episode_size = episode_size
        self.move_range = move_range
        self.img_size = img_size
        self.batch_size = batch_size
        self.device = device
        self.logger = logger
        self.global_avg_test_rewards = global_avg_test_rewards

        self.state = State((self.batch_size, 1, self.img_size[0], self.img_size[1]), self.move_range)

    def run(self):
        self.agent.process_idx = self.process_idx
        print(f"[{self.process_idx}] Start Test")

        # input image
        raw_x = self.X.cpu().numpy()
        # print(f"X shape: {raw_x.shape}")
        # generate noise
        raw_n = np.zeros(raw_x.shape, dtype=np.float32)
        # raw_n = np.random.normal(0, 15, raw_x.shape).astype(raw_x.dtype)/255
         # initialize the current state and reward
        self.state.reset(raw_x, raw_n)
        self.agent.clear_memory()
        for t in range(0, self.episode_size):
            action = self.agent.act(torch.from_numpy(self.state.image).to(self.device))
            self.state.step(action)

        image_words = [read_image_array_words(self.state.image[b, 0]) for b in range(self.state.image.shape[0])]
        reading_rewards = [ 100 / (compare_strings_levenshtein(image_words[b], self.y[b]) + 1) for b in range(len(self.y))]
        reward = [r*np.power(self.gamma, t) for r in reading_rewards]
        avg_reward = np.mean(reward)

        print(f"[{self.process_idx}] Test avg reward: {avg_reward}")
        with self.global_avg_test_rewards.get_lock():
            self.global_avg_test_rewards[self.process_idx] = avg_reward
