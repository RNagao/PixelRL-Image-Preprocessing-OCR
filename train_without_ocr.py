import os
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
# from src.train import Trainer
import torch.optim as optim
from src.train import train_pixelwise_reward

from src.utils import save_model

# Hyperparams
IMG_SIZE = (63, 63)
BATCH_SIZE = 128
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

MODEL_NAME = f"pixelrl_upLR_{N_EPISODES}eps_{EPISODE_SIZE}steps_{LEARNING_RATE}lr_{GAMMA}gamma"
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
    fcn = FCN(n_actions=N_ACTIONS,
               num_channels=INPUT_SHAPE,
               hidden_units=HIDDEN_UNITS).to(device)
    fcn.load_state_dict(torch.load("./torch_initweight/sig25_gray.pth"))
    
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

    # train
    print(f"\nTRAINNING DEVICE: {device}")
    print(f"NUM_WORKERS: {NUM_WORKERS}")
    print(f"TRAIN DATALOADER SIZE: {len(train_dataloader)}")

    fcn.train()

    train_start = time.time()
    fcn, ep_load = load_checkpoint(TARGET_DIR, fcn, device)

    rewards = []
    losses = []

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    plt.title(MODEL_NAME)
    plt.ion()

    for ep in tqdm(range(ep_load, N_EPISODES), desc="EPISODES", initial=ep_load, total=N_EPISODES):
        ep_start = time.time()
        for b, (X, y) in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="DATALOADER"):
            reward, loss = train_pixelwise_reward(process_idx=b,
                    model=fcn,
                    optimizer=optimizer,
                    X=X,
                    n_episodes=ep,
                    episode_size=EPISODE_SIZE,
                    lr=LEARNING_RATE,
                    gamma=GAMMA,
                    move_range=MOVE_RANGE,
                    img_size=IMG_SIZE,
                    batch_size=BATCH_SIZE,
                    device=device,
                    logger=None,
                    model_hidden_units=HIDDEN_UNITS)
            rewards.append(reward.item())
            losses.append(loss.item())

            if len(rewards) > 500:
                rewards.pop(0)
                losses.pop(0)

            atualizar_graficos(ax1, ax2, rewards, losses)

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
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor()
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


def atualizar_graficos(ax1, ax2, rewards, losses):
    ax1.clear()
    ax2.clear()
    
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