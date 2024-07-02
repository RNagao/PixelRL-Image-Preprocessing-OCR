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

torch.autograd.set_detect_anomaly(True)
def train(
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

    agent = PixelWiseAgent(model=model,
                        optimizer=optimizer,
                        lr=lr,
                        t_max=episode_size,
                        gamma=gamma,
                        batch_size=batch_size,
                        img_size=img_size,
                        device=device,
                        logger=logger)

    state = State((batch_size, 1, img_size[0], img_size[1]), move_range, model_hidden_units=model_hidden_units)

    print(f"[{process_idx}] Start Train")

    # input image
    raw_x = X.cpu().numpy()
    # print(f"X shape: {raw_x.shape}")
    # generate noise
    raw_n = np.zeros(raw_x.shape, dtype=np.float32)
    # raw_n = np.random.normal(0, 15, raw_x.shape).astype(raw_x.dtype)/255
    # initialize the current state and reward
    state.reset(raw_x, raw_n)
    agent.clear_memory()

    sum_reward = 0
    reward = np.zeros(len(raw_x), raw_x.dtype)
    prev_dist = calculate_levenshtein_dist(state.image, y)

    for t in range(0, episode_size):
        print(f"[{process_idx}] Episode {n_episodes} step {t}")

        current_image_tensor = state.tensor
        action, inner_state, action_prob = agent.act_and_train(current_image_tensor, reward)

        state.step(action, inner_state)

        # reward = _calculate_reward(reward, state.image)
        reward, prev_dist = calculate_reward(prev_dist, state.image, y)

        sum_reward += np.mean(reward) * np.power(gamma, t)

    agent.stop_episode_and_train(current_image_tensor, reward, True, process_idx=process_idx)
    # agent.optimizer.lr = lr*((1-episode/n_episodes)**0.9)
    print(f"[{process_idx}] Train total reward: {sum_reward}")


    with global_avg_train_rewards.get_lock():
        global_avg_train_rewards[process_idx] += sum_reward


def calculate_reward(prev_dist, current_image, y):
    levenshtein = calculate_levenshtein_dist(current_image, y)
    reward = prev_dist - levenshtein
    return reward, levenshtein

def calculate_levenshtein_dist(current_image, y):
    # print(f"LOAD BALANCE: {running_processes[:]}")
    # dist = None
    # while dist is None:
    #     if running_processes[0] < 3:
    #         with running_processes.get_lock():
    #             running_processes[0] += 1
    #         dist = _local_levenshtein_dist(current_image)
    #         with running_processes.get_lock():
    #             running_processes[0] -= 1
    #     elif running_processes[1] < 3:
    #         with running_processes.get_lock():
    #             running_processes[1] += 1
    #         dist = _api_koyeb_levenshtein_dist(current_image)
    #         with running_processes.get_lock():
    #             running_processes[1] -= 1
    #     elif running_processes[2] < 5:
    #         with running_processes.get_lock():
    #             running_processes[2] += 1
    #         dist = _lambda_levenshtein_dist(current_image)
    #         with running_processes.get_lock():
    #             running_processes[2] -= 1
    return local_levenshtein_dist(current_image, y)

# def _api_koyeb_levenshtein_dist(current_image):
#     url = os.getenv('KOYEB_URL') or ''

#     levenshtein_dists = []
#     array_size = 4
#     if len(y) % array_size != 0:
#         return None

#     for i in range(len(y) // array_size):
#         value = []
#         while len(value) != array_size:
#             try:
#                 with requests.post(url, json={
#                     "image_array": current_image.tolist()[i*array_size:i*array_size+array_size],
#                     "text_assert": list(y)[i*array_size:i*array_size+array_size]
#                 }) as r:
#                     value = r.json().get('result')
#             except requests.exceptions.ChunkedEncodingError as e:
#                 print(e)
#             except requests.exceptions.JSONDecodeError as e:
#                 print(e)
#         levenshtein_dists.extend(value)
    
#     if len(levenshtein_dists) != len(y):
#         return None

#     return np.array(levenshtein_dists)

def local_levenshtein_dist(current_image, y):
    image_words = [read_image_array_words(current_image[b, 0]) for b in range(len(current_image))]
    levenshtein = [compare_strings_levenshtein(image_words[b], y[b]) for b in range(len(current_image))]
    return np.array(levenshtein)

# def _lambda_levenshtein_dist(current_image):
#     if lambda_client is None:
#         lambda_client = LambdaClient()
    
#     payload = {
#         "httpMethod": "POST",
#         "path": "/levenshtein_dist",
#         "queryStringParameters": {},
#         "headers": {
#             "content_type": "application/json"
#         },
#         "data": {
#             "image_array": current_image.tolist(),
#             "text_assert": list(y)
#         }
#     }
#     result = lambda_client.invoke_lambda(payload)
#     return np.array(result.get('result'))