import torch
import numpy as np
import logging
import torch.multiprocessing as mp
from tqdm.auto import tqdm
from typing import Tuple
from torch import nn

from src.agent import PixelWiseAgent
from src.state import State
from src.reader import compare_strings_levenshtein, read_image_array_words

class Trainer(mp.Process):
    def __init__(self,
                process_idx: int,
                agent: PixelWiseAgent,
                # optimizer: torch.optim.Optimizer,
                X,
                y,
                n_episodes: int,
                episode_size: int,
                lr: int,
                gamma: float,
                move_range:int,
                logger:logging.Logger,
                img_size:tuple[int,int]=(481, 321),
                batch_size:int=32,
                device:torch.device="cpu"):
        super().__init__()

        self.process_idx = process_idx
        self.agent = agent
        # self.optimizer = optimizer
        self.X = X
        self.y = y
        self.n_episodes = n_episodes
        self.episode_size = episode_size
        self.lr = lr
        self.gamma = gamma
        self.move_range = move_range
        self.img_size = img_size
        self.batch_size = batch_size
        self.device = device
        self.logger = logger

        self.state = State((self.batch_size, 1, self.img_size[0], self.img_size[1]), self.move_range)

    def run(self):
        self.process_idx = self.process_idx
        print(f"[{self.process_idx}] Start Train")

        for episode in range(self.n_episodes):
            # input image
            raw_x = self.X.cpu().numpy()
            # print(f"X shape: {raw_x.shape}")
            # generate noise
            raw_n = np.zeros(raw_x.shape, dtype=np.float32)
            # raw_n = np.random.normal(0, 15, raw_x.shape).astype(raw_x.dtype)/255
            # initialize the current state and reward
            self.state.reset(raw_x, raw_n)
            avg_max_reward = 0.0
            self.agent.clear_memory()
            for t in range(0, self.episode_size):
                print(f"[{self.process_idx}] Episode {episode} step {t}")
                reward = self._calculate_reward(t)
                avg_max_reward = torch.mean(reward) if torch.mean(reward) > avg_max_reward else avg_max_reward
                action = self.agent.act_and_train(torch.from_numpy(self.state.image).to(self.device), reward, process_idx=self.process_idx)
                self.state.step(action)

            self.agent.stop_episode_and_train(torch.from_numpy(self.state.image).to(self.device), reward, True, process_idx=self.process_idx)
            # self.agent.optimizer.lr = self.lr*((1-episode/self.n_episodes)**0.9)
            print(f"[{self.process_idx}] Local train avg max reward: {avg_max_reward}")
            self.agent.update_train_avg_reward(avg_max_reward)
            print(f"[{self.process_idx}] Global train avg max reward: {self.agent.get_train_avg_reward()}")

    def _calculate_reward(self, t):
        image_words = [read_image_array_words(self.state.image[b, 0]) for b in range(self.batch_size)]
        reading_rewards = [ 100 / (compare_strings_levenshtein(image_words[b], self.y[b]) + 1) for b in range(self.batch_size)]
        reward = [r*np.power(self.gamma, t) for r in reading_rewards]
        return torch.tensor(reward, device=self.device)
