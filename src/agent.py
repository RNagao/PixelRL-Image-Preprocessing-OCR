import torch
import copy
import logging
from torch.distributions import Categorical
from src.models import FCN

class PixelWiseAgent():
    def __init__(self,
                model:FCN,
                optimizer: torch.optim.Optimizer,
                t_max:int,
                gamma:float,
                logger:logging.Logger,
                beta:float=1e-2,
                pi_loss_coef:float=1.0,
                v_loss_coef:float=0.5,
                keep_loss_scale_same:bool=False,
                normalize_grad_by_t_max:bool=False,
                use_average_reward:bool=False,
                average_reward_tau:float=1e-2,
                act_deterministically:bool=False,
                average_entropy_decay:int=0.999,
                average_value_decay:int=0.999,
                batch_size:int=32,
                img_size:tuple[int,int]=(481, 321),
                device="cpu"):
        self.shared_model = model
        self.model = copy.deepcopy(self.shared_model)
        self.optimizer = optimizer
        self.t_max = t_max
        self.gamma = gamma
        self.beta = beta
        self.pi_loss_coef = pi_loss_coef
        self.v_loss_coef = v_loss_coef
        self.keep_loss_scale_same = keep_loss_scale_same
        self.normalize_grad_by_t_max = normalize_grad_by_t_max
        self.use_average_reward = use_average_reward
        self.average_reward_tau = average_reward_tau
        self.act_deterministically = act_deterministically
        self.average_value_decay = average_value_decay
        self.average_entropy_decay = average_entropy_decay
        self.batch_size=batch_size
        self.img_size=img_size
        self.device = device

        self.t = 0
        self.t_start = 0
        self.past_action_log_prob = {}
        self.past_action_entropy = {}
        self.past_states = {}
        self.past_rewards = {}
        self.past_values = {}
        self.loss_tracking = []
        self.average_reward = 0

        # Stats
        self.average_value = 0.0
        self.average_entropy = 0.0

        self.logger = logger

    def clear_memory(self):
        self.past_action_log_prob = {}
        self.past_action_entropy = {}
        self.past_states = {}
        self.past_rewards = {}
        self.past_values = {}

    def update(self, state_var, process_idx=0):
        assert self.t_start < self.t

        if state_var is None:
            R = torch.zeros(size=(self.batch_size, 1, self.img_size[0], self.img_size[1]), requires_grad=True, device=self.device)
        else:
            print(f"[{process_idx}] State var update")
            self.model.eval()
            with torch.inference_mode():
                _, vout = self.model(state_var)
                R = vout.requires_grad_(True).type(torch.float32).to(self.device)

        pi_loss = 0
        v_loss = 0
        self.model.train()
        for i in reversed(range(self.t_start, self.t)):
            R = R * self.gamma
            past_reward = self.past_rewards[i]
            for b in range(self.batch_size):
                R[b,0] = torch.add(R[b,0], past_reward[b])
            if self.use_average_reward:
                R = R - self.average_reward
            v = self.past_values[i]
            advantage = R - v
            if self.use_average_reward:
                self.average_reward += self.average_reward_tau * float(advantage)

            # Accumulate gradients of policy
            log_prob = self.past_action_log_prob[i].unsqueeze(dim=1)
            entropy = self.past_action_entropy[i].unsqueeze(dim=1)

            # Log probability is increased proportinally to advantage
            pi_loss -= log_prob * advantage.type(torch.float32)

            # Entropy is maximized
            pi_loss -= self.beta * entropy

            # Accumulate gradients of value function
            v_loss += (v - R) ** 2 / 2

        if self.pi_loss_coef != 1.0:
            pi_loss *= self.pi_loss_coef

        if self.v_loss_coef != 1.0:
            v_loss *= self.v_loss_coef

        # Normalize the loss of sequences truncated by terminal states
        if self.keep_loss_scale_same and self.t - self.t_start < self.t_max:
            factor = self.t_max / (self.t - self.t_start)
            pi_loss *= factor
            v_loss *= factor

        if self.normalize_grad_by_t_max:
            pi_loss /= self.t - self.t_start
            v_loss /= self.t - self.t_start

        # if process_idx == 0:
        #   print(f"\npi_loss:\n{pi_loss}\n\nv_loss:\n{v_loss}")

        total_loss = torch.mean(pi_loss + v_loss.reshape(pi_loss.shape))
        self.loss_tracking.append(total_loss)
        print(f"[{process_idx}] Loss: {total_loss}")

        # Compute gradients using thread-specific model
        self.model.zero_grad()
        self.shared_model.zero_grad()
        self.optimizer.zero_grad()
        total_loss.backward()

        for local_params, global_params in zip(self.model.parameters(), self.shared_model.parameters()):
            global_params._grad = local_params.grad


        self.optimizer.step()
        # if process_idx == 0:
        print(f'[{process_idx}] Update')

        self.clear_memory()

        self.t_start = self.t

    def act_and_train(self, state_var, reward, process_idx=0):
        # print(f"[{process_idx}] Act and train\nt: {self.t} | t_start: {self.t_start} | t_max: {self.t_max}")
        self.past_rewards[self.t-1] = reward
        
        if self.t - self.t_start == self.t_max:
            self.update(state_var, process_idx=process_idx)

        self.past_states[self.t] = state_var

        self.model.eval()
        with torch.inference_mode():
            pout, vout = self.model(state_var)

        dist = Categorical(pout.permute([0, 2, 3, 1]))
        action = dist.sample()

        self.past_action_log_prob[self.t] = dist.log_prob(action)
        self.past_action_entropy[self.t] = dist.entropy()
        self.past_values[self.t] = vout

        self.t += 1

        return action.cpu().numpy()

    def act(self, obs):
        self.model.eval()
        with torch.inference_mode():
            state_var = obs.to(self.device)
            pout, _ = self.model(state_var)
            if self.act_deterministically:
                return torch.argmax(pout).cpu().numpy()
            else:
                dist = Categorical(pout.permute([0, 2, 3, 1]))
                return dist.sample().cpu().numpy()

    def stop_episode_and_train(self, state_var, reward, done=False, process_idx=0):
        print(f'[{process_idx}] Stop and Train')

        self.past_rewards[self.t-1] =reward
        if done:
            self.update(None, process_idx=process_idx)
        else:
            self.update(state_var, process_idx=process_idx)

    def get_statistics(self):
        return {
            "average_value": self.average_value,
            "average_entropy": self.average_entropy
        }

    def update_train_avg_reward(self, r):
        self.shared_model.update_train_avg_reward(r)

    def update_test_avg_reward(self, r):
        self.shared_model.update_test_avg_reward(r)

    def get_train_avg_reward(self):
        return self.shared_model.train_average_max_reward

    def get_test_avg_reward(self):
        return self.shared_model.test_average_max_reward