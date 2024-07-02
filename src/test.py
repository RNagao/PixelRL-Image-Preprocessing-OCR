import torch
import numpy as np
import logging
import torch.multiprocessing as mp
from tqdm.auto import tqdm
from typing import Tuple
from statistics import fmean
from torch.distributions import Categorical

from src.agent import PixelWiseAgent
from src.state import State
from src.reader import compare_strings_levenshtein, read_image_array_words


def test(model: torch.nn.Module,
            X,
            y,
            episode_size: int,
            move_range:int,
            model_hidden_units:int=64,
            act_deterministically:bool=False,
            device:torch.device="cpu"):

    state = State(X.shape, move_range, model_hidden_units)

    # input image
    raw_x = X.cpu().numpy()
    # print(f"X shape: {raw_x.shape}")
    # generate noise
    raw_n = np.zeros(raw_x.shape, dtype=np.float32)
    # raw_n = np.random.normal(0, 15, raw_x.shape).astype(raw_x.dtype)/255
        # initialize the current state and reward
    state.reset(raw_x, raw_n)
    for t in range(0, episode_size):
        action, inner_state = select_action(torch.from_numpy(state.tensor).to(device), model, device, act_deterministically)
        state.step(action, inner_state)

    return calculate_reward(state.image, y)

def calculate_reward(current_image, y):
    image_words = [read_image_array_words(current_image[b, 0]) for b in range(len(current_image))]
    levenshtein = [compare_strings_levenshtein(image_words[b], y[b]) for b in range(len(current_image))]
    avg_reward = np.mean(levenshtein)
    return avg_reward

def select_action(obs, model, device, act_deterministically):
    model.eval()
    with torch.inference_mode():
        state_var = obs.to(device)
        pout, _, inner_state = model.pi_and_v(state_var)
        p_trans = pout.permute([0, 2, 3, 1])
        dist = Categorical(p_trans)
        if act_deterministically:
            return torch.argmax(pout.detach()).cpu().numpy(), inner_state.detach().cpu()
        else:
            dist = Categorical(pout.permute([0, 2, 3, 1]))
            return dist.sample().detach().cpu().numpy(), inner_state.detach().cpu()