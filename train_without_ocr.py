import os
import re
import torch
import torch.multiprocessing as mp
import numpy as np
import time
from torchinfo import summary
from pathlib import Path
from tqdm import tqdm
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.animation as animation

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
from src.models import *
from src.agent import PixelWiseAgent
from src.share_optim import SharedAdam
from torch.utils.data import DataLoader, ConcatDataset
from src.dataset import ImageCustomDataset
from src.agent import PixelWiseAgent, PixelWiseAgentWithoutOCR
from src.state import State
# from src.train import Trainer
import torch.optim as optim
from src.train import train_pixelwise_reward

from src.utils import save_model

# Hyperparams
IMG_SIZE = (70, 70)
BATCH_SIZE = 32
NUM_WORKERS = 1
NUM_WORKERS = os.cpu_count() - 1
# NUM_WORKERS = 30

INPUT_SHAPE = 1
N_ACTIONS = 9
MOVE_RANGE = 3 # number of actions that move pixel values
HIDDEN_UNITS = 64
OUTPUT_SHAPE = INPUT_SHAPE

LEARNING_RATE = 0.001
GAMMA = 0.95
EPISODE_SIZE= 5
N_EPISODES = 30000

MODEL_NAME = f"pixelrl_2_{N_EPISODES}eps_{EPISODE_SIZE}steps_{LEARNING_RATE}lr_{GAMMA}gamma"
TARGET_DIR = f"./models/{MODEL_NAME}"

def main():

    # device agnostic code
    # device = "cpu"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    datasets_paths = [
        "datasets/BSD68/train/",
        "datasets/Waterloo/train/",
        "datasets/SROIE2019/train/img",
    ]
    # Create dataloaders
    train_dataloader = create_train_dataset(datasets_paths)

    # create model
    fcn = FCN2(n_actions=N_ACTIONS,
               num_channels=INPUT_SHAPE,
               hidden_units=HIDDEN_UNITS).to(device)
    load_parameters_from_npz(fcn, "./torch_initweight/pretrained_15.npz")
    # fcn.load_state_dict(torch.load("./torch_initweight/sig25_gray.pth"))
    
    # fcn.share_memory()

    print("\n\nMODEL SUMMARY")
    # summary(model=fcn,
    #     input_size=(1, 1, IMG_SIZE[0], IMG_SIZE[1]),
    #     col_names=["input_size", "output_size", "num_params", "trainable"],
    #     # col_width=20,
    #     row_settings=["var_names"],
    #     device=device)

    # setup optimizer
    optimizer = optim.Adam(params=fcn.parameters(), lr=LEARNING_RATE)
    # optimizer = SharedAdam(params=fcn.parameters(), lr=LEARNING_RATE)
    # optimizer.share_memory()

    # setup agent
    agent = PixelWiseAgentWithoutOCR(model=fcn,
                        optimizer=optimizer,
                        lr=LEARNING_RATE,
                        t_max=EPISODE_SIZE,
                        gamma=GAMMA,
                        batch_size=BATCH_SIZE,
                        img_size=IMG_SIZE,
                        device=device,
                        logger=None
                        )

    state = State((BATCH_SIZE, 1, IMG_SIZE[0], IMG_SIZE[1]), MOVE_RANGE, model_hidden_units=HIDDEN_UNITS)

    # train
    print(f"\nTRAINNING DEVICE: {device}")
    print(f"NUM_WORKERS: {NUM_WORKERS}")
    print(f"TRAIN DATALOADER SIZE: {len(train_dataloader)}")

    fcn.train()

    train_start = time.time()
    fcn, ep_load = load_checkpoint(TARGET_DIR, fcn, device)

    rewards = []
    losses = []
    rewards_ep_dict = {
        "high": [],
        "lower": [],
        "mean": []
    }
    losses_ep_dict = {
        "high": [],
        "lower": [],
        "mean": []
    }

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    ax1, ax2, ax3, ax4 = axs.flatten()
    plt.title(MODEL_NAME)
    plt.ion()
    plt.grid(True)

    torch.cuda.empty_cache()
    for ep in tqdm(range(ep_load, N_EPISODES), desc="EPISODES", initial=ep_load, total=N_EPISODES):
        ep_start = time.time()
        ep_rewards = []
        ep_losses = []
        for b, (X, y) in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="DATALOADER"):
            reward, loss = train_pixelwise_reward(process_idx=b,
                                                    agent=agent,
                                                    state=state,
                                                    X=X,
                                                    episode_size=EPISODE_SIZE,
                                                    gamma=GAMMA,
                                                    device=device,
                                                    )
            rewards.append(reward.item() * 255)
            losses.append(loss.item())
            ep_rewards.append(reward.item() * 255)
            ep_losses.append(loss.item())

            if len(rewards) > 300:
                rewards.pop(0)
                losses.pop(0)

            atualizar_graficos(ax1, ax2, ax3, ax4, rewards, losses, rewards_ep_dict, losses_ep_dict)

        rewards_ep_dict["high"].append(max(ep_rewards))
        rewards_ep_dict["lower"].append(min(ep_rewards))
        rewards_ep_dict["mean"].append(sum(ep_rewards)/len(ep_rewards))
        losses_ep_dict["high"].append(max(ep_losses))
        losses_ep_dict["lower"].append(min(ep_losses))
        losses_ep_dict["mean"].append(sum(ep_losses)/len(ep_losses))

        print(f"EP train time: {time.time() - ep_start}")
        update_learning_rate(optimizer, ep, N_EPISODES, LEARNING_RATE)
        if ep % 10 == 0:
            save_model(model=fcn,
                target_dir=TARGET_DIR,
                model_name=f"checkpoint_{ep}.pth")
            print(f"SAVED CHECKPOINT {ep}")

    train_stop = time.time()

    plt.ioff()
    plt.show()

    print(f"\nTRAIN TIME: {train_stop - train_start}")

    save_model(model=fcn,
               target_dir=TARGET_DIR,
               model_name=f"{MODEL_NAME}.pth")
    

def load_checkpoint(target_dir, model, device):
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

    model.load_state_dict(torch.load(last_checkpoint, map_location=torch.device(device)))

    return model, i


def create_train_dataset(datasets_path_dir):
    transforms_list = transforms.Compose([
        transforms.RandomCrop(IMG_SIZE),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    train_datasets = []
    for dir_path in datasets_path_dir:
        train_datasets.append(ImageCustomDataset(
            images_dir=dir_path,
            asserts_dir=None,
            transform=transforms_list
        ))
    
    dataset = ConcatDataset(train_datasets)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            num_workers=NUM_WORKERS)
    return dataloader

def load_parameters_from_npz(model, npz_file):
    # Carregar parâmetros do arquivo .npz
    data = np.load(npz_file)
    state_dict = model.state_dict()

    for key in data.keys():
        model_key = re.sub(r"\/diconv", "", key)
        model_key = re.sub(r"\/model", ".0", model_key)
        model_key = re.sub(r"\/W", ".weight", model_key)
        model_key = re.sub(r"\/b", ".bias", model_key)

        if model_key in state_dict:
            # Converter o numpy array para tensor e atribuir ao estado do modelo
            state_dict[model_key] = torch.from_numpy(data[key])
        else:
            print(f"model has no key {model_key} / {key}")
    
    # Atualizar o estado do modelo
    model.load_state_dict(state_dict)

def atualizar_graficos(ax1, ax2, ax3, ax4, rewards, losses, rewards_ep_dict, losses_ep_dict):
    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax4.clear()

    # Atualiza o gráfico de recompensas
    ax1.plot(rewards, label='Recompensa', color='blue')
    ax1.set_xlabel('Iteração')
    ax1.set_ylabel('Recompensa')
    ax1.set_title('Recompensa em Tempo Real')
    ax1.legend()
    
    # Atualiza o gráfico de erros
    ax2.plot(losses, label='Erro', color='red')
    ax2.set_xlabel('Iteração')
    ax2.set_ylabel('Erro')
    ax2.set_title('Erro em Tempo Real')
    ax2.legend()

    ax3.plot(rewards_ep_dict["high"], label='Maiores Recompensa', color='green')
    ax3.plot(rewards_ep_dict["lower"], label='Menores Recompensa', color='red')
    ax3.plot(rewards_ep_dict["mean"], label='Media Recompensas', color='blue')
    ax3.set_xlabel('Iteração')
    ax3.set_ylabel('Recompensa')
    ax3.set_title('Recompensa por Ep')
    ax3.legend()

    ax4.plot(losses_ep_dict["high"], label='Maiores Erro', color='red')
    ax4.plot(losses_ep_dict["lower"], label='Menores Erro', color='green')
    ax4.plot(losses_ep_dict["mean"], label='Media Erros', color='blue')
    ax4.set_xlabel('Iteração')
    ax4.set_ylabel('Erro')
    ax4.set_title('Erro por Ep')
    ax4.legend()
    
    plt.draw()
    plt.pause(0.01)  # Pausa para atualizar o gráfico

def update_learning_rate(optimizer, episode, total_episodes, initial_lr):
    """Atualiza a taxa de aprendizado conforme a política polinomial."""
    # Calcula o fator de decaimento
    lr = initial_lr * (1 - episode / total_episodes) ** 0.9
    
    # Atualiza a taxa de aprendizado do otimizador
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == "__main__":
    main()