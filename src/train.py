import torch
import numpy as np
import logging
import torch.multiprocessing as mp
from statistics import fmean
from tqdm.auto import tqdm
from typing import Tuple
from torch import nn

from src.agent import PixelWiseAgent
from src.state import State
from src.reader import compare_strings_levenshtein, read_image_array_words

class Trainer(mp.Process):
    torch.autograd.set_detect_anomaly(True)

    def __init__(self,
                process_idx: int,
                model: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                X,
                y,
                n_episodes: int,
                episode_size: int,
                lr: int,
                gamma: float,
                move_range:int,
                global_avg_train_rewards,
                logger:logging.Logger,
                img_size:tuple[int,int]=(481, 321),
                batch_size:int=32,
                model_hidden_units:int=64,
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
        self.global_avg_train_rewards = global_avg_train_rewards
        self.device = device
        self.logger = logger

        self.state = State((self.batch_size, 1, self.img_size[0], self.img_size[1]), self.move_range, model_hidden_units=model_hidden_units)

    def run(self):
        self.process_idx = self.process_idx
        print(f"[{self.process_idx}] Start Train")

        # input image
        raw_x = self.X.cpu().numpy()
        # print(f"X shape: {raw_x.shape}")
        # generate noise
        raw_n = np.zeros(raw_x.shape, dtype=np.float32)
        # raw_n = np.random.normal(0, 15, raw_x.shape).astype(raw_x.dtype)/255
        # initialize the current state and reward
        self.state.reset(raw_x, raw_n)
        self.agent.clear_memory()

        sum_reward = 0
        reward = np.zeros(len(raw_x), raw_x.dtype)

        for t in range(0, self.episode_size):
            print(f"[{self.process_idx}] Episode {self.n_episodes} step {t}")

            current_image_tensor = self.state.tensor
            action, inner_state, action_prob = self.agent.act_and_train(current_image_tensor, reward)

            self.state.step(action, inner_state)

            reward = self._calculate_reward(reward, self.state.image)

            sum_reward += np.mean(reward) * np.power(self.gamma, t)

        self.agent.stop_episode_and_train(current_image_tensor, reward, True, process_idx=self.process_idx)
        # self.agent.optimizer.lr = self.lr*((1-episode/self.n_episodes)**0.9)
        print(f"[{self.process_idx}] Train total reward: {sum_reward}")


        # with self.global_avg_train_rewards.get_lock():
        #     self.global_avg_train_rewards[self.process_idx] = ep_final_image_total_reward


    def _calculate_reward(self, prev_reward, current_image):
        image_words = [read_image_array_words(current_image[b, 0]) for b in range(len(current_image))]
        inverse_levenshtein = [compare_strings_levenshtein(image_words[b], self.y[b]) for b in range(len(current_image))]
        reward = np.absolute(prev_reward) - np.absolute(np.array(inverse_levenshtein, prev_reward.dtype))
        return reward
