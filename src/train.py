import torch
import numpy as np
import logging
import torch.multiprocessing as mp
import requests
import itertools
import os, signal
from tqdm.auto import tqdm
from torch import nn

from src.agent import PixelWiseAgent
from src.state import State
from src.reader import compare_strings_levenshtein, read_image_array_words
from src.lambda_client import LambdaClient

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
                running_processes,
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
        self.lambda_client = None
        self.running_processes = running_processes

    def run(self):
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
        prev_dist = self.calculate_levenshtein_dist(self.state.image)

        for t in range(0, self.episode_size):
            print(f"[{self.process_idx}] Episode {self.n_episodes} step {t}")

            current_image_tensor = self.state.tensor
            action, inner_state, action_prob = self.agent.act_and_train(current_image_tensor, reward)

            self.state.step(action, inner_state)

            # reward = self._calculate_reward(reward, self.state.image)
            reward, prev_dist = self._calculate_reward(prev_dist, self.state.image)

            sum_reward += np.mean(reward) * np.power(self.gamma, t)

        self.agent.stop_episode_and_train(current_image_tensor, reward, True, process_idx=self.process_idx)
        # self.agent.optimizer.lr = self.lr*((1-episode/self.n_episodes)**0.9)
        print(f"[{self.process_idx}] Train total reward: {sum_reward}")


        with self.global_avg_train_rewards.get_lock():
            self.global_avg_train_rewards[self.process_idx] += sum_reward


    def _calculate_reward(self, prev_dist, current_image):
        levenshtein = self.calculate_levenshtein_dist(current_image)
        reward = prev_dist - levenshtein
        return reward, levenshtein

    def calculate_levenshtein_dist(self, current_image):
        print(f"LOAD BALANCE: {self.running_processes[:]}")
        dist = None
        while dist is None:
            if self.running_processes[0] < 3:
                with self.running_processes.get_lock():
                    self.running_processes[0] += 1
                dist = self._local_levenshtein_dist(current_image)
                with self.running_processes.get_lock():
                    self.running_processes[0] -= 1
            elif self.running_processes[1] < 3:
                with self.running_processes.get_lock():
                    self.running_processes[1] += 1
                dist = self._api_koyeb_levenshtein_dist(current_image)
                with self.running_processes.get_lock():
                    self.running_processes[1] -= 1
            elif self.running_processes[2] < 5:
                with self.running_processes.get_lock():
                    self.running_processes[2] += 1
                dist = self._lambda_levenshtein_dist(current_image)
                with self.running_processes.get_lock():
                    self.running_processes[2] -= 1
        return dist

    def _api_koyeb_levenshtein_dist(self, current_image):
        url = os.getenv('KOYEB_URL') or ''

        levenshtein_dists = []
        array_size = 4
        if len(self.y) % array_size != 0:
            return None

        for i in range(len(self.y) // array_size):
            value = []
            while len(value) != array_size:
                try:
                    with requests.post(url, json={
                        "image_array": current_image.tolist()[i*array_size:i*array_size+array_size],
                        "text_assert": list(self.y)[i*array_size:i*array_size+array_size]
                    }) as r:
                        value = r.json().get('result')
                except requests.exceptions.ChunkedEncodingError as e:
                    print(e)
                except requests.exceptions.JSONDecodeError as e:
                    print(e)
            levenshtein_dists.extend(value)
        
        if len(levenshtein_dists) != len(self.y):
            return None

        return np.array(levenshtein_dists)
    
    def _local_levenshtein_dist(self, current_image):
        image_words = [read_image_array_words(current_image[b, 0]) for b in range(len(current_image))]
        levenshtein = [compare_strings_levenshtein(image_words[b], self.y[b]) for b in range(len(current_image))]
        return np.array(levenshtein)
    
    def _lambda_levenshtein_dist(self, current_image):
        if self.lambda_client is None:
            self.lambda_client = LambdaClient()
        
        payload = {
            "httpMethod": "POST",
            "path": "/levenshtein_dist",
            "queryStringParameters": {},
            "headers": {
                "content_type": "application/json"
            },
            "data": {
                "image_array": current_image.tolist(),
                "text_assert": list(self.y)
            }
        }
        result = self.lambda_client.invoke_lambda(payload)
        return np.array(result.get('result'))