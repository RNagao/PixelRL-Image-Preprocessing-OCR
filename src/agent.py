import torch
import copy
import logging
import numpy as np
from torch.distributions import Categorical
from torch.autograd import Variable

class PixelWiseAgent():
    def __init__(self,
                model:torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                t_max:int,
                gamma:float,
                lr: float,
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
        # self.model = copy.deepcopy(self.shared_model)
        self.optimizer = optimizer
        self.t_max = t_max
        self.lr = lr
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

    # def update_grad(self, target, source):
    #     target_params = dict(target.named_parameters())
    #     # print(target_params)
    #     for param_name, param in source.named_parameters():
    #         if target_params[param_name].grad is None:
    #             if param.grad is None:
    #                 pass
    #             else:
    #                 target_params[param_name].grad = param.grad
    #         else:
    #             if param.grad is None:
    #                 target_params[param_name].grad = None
    #             else:
    #                 target_params[param_name].grad[...] = param.grad


    # def sync_parameters(self):
    #     for m1, m2 in zip(self.model.modules(), self.shared_model.modules()):
    #         m1._buffers = m2._buffers.copy()
    #     for target_param, param in zip(self.model.parameters(), self.shared_model.parameters()):
    #         target_param.detach().copy_(param.detach())

    def clear_memory(self):
        self.past_action_log_prob = {}
        self.past_action_entropy = {}
        self.past_states = {}
        self.past_rewards = {}
        self.past_values = {}

    def update(self, state_var, batch_size, process_idx=0):
        assert self.t_start < self.t

        if state_var is None:
            R = torch.zeros(size=(batch_size, 1, self.img_size[0], self.img_size[1]), device=self.device)
        else:
            print(f"[{process_idx}] State var update")
            _, vout = self.shared_model.pi_and_v(state_var)
            R = vout.detach().to(self.device)

        pi_loss = 0
        v_loss = 0
        for i in reversed(range(self.t_start, self.t)):
            R *= self.gamma
            reward = self.past_rewards[i]
            R += torch.from_numpy(reward[:, np.newaxis, np.newaxis, np.newaxis]).to(self.device)
            if self.use_average_reward:
                R = R - self.average_reward
            v = self.past_values[i]
            advantage = R - v.detach()
            if self.use_average_reward:
                self.average_reward += self.average_reward_tau * float(advantage.detach())

            # Accumulate gradients of policy
            log_prob = self.past_action_log_prob[i]
            entropy = self.past_action_entropy[i]

            # Log probability is increased proportinally to advantage
            pi_loss -= log_prob * advantage.detach()

            # Entropy is maximized
            pi_loss -= self.beta * entropy

            # Accumulate gradients of value function
            v_loss += (v - R) ** 2 / 2

        if self.pi_loss_coef != 1.0:
            pi_loss = pi_loss * self.pi_loss_coef

        if self.v_loss_coef != 1.0:
            v_loss = v_loss * self.v_loss_coef

        # Normalize the loss of sequences truncated by terminal states
        if self.keep_loss_scale_same and self.t - self.t_start < self.t_max:
            factor = self.t_max / (self.t - self.t_start)
            pi_loss = pi_loss * factor
            v_loss = v_loss * factor

        if self.normalize_grad_by_t_max:
            pi_loss = pi_loss/(self.t - self.t_start)
            v_loss = v_loss/(self.t - self.t_start)

        # if process_idx == 0:
        #   print(f"\npi_loss:\n{pi_loss}\n\nv_loss:\n{v_loss}")

        total_loss = (pi_loss + v_loss).mean()
        #self.loss_tracking.append(total_loss)
        print(f"[{process_idx}] Loss: {total_loss}")

        # Compute gradients using thread-specific model
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # self.update_grad(self.shared_model, self.model)
        # self.sync_parameters()

        print(f'[{process_idx}] Update')

        # self.model.load_state_dict(self.shared_model.state_dict())
        self.clear_memory()

        self.t_start = self.t

    def act_and_train(self, state_var, reward, process_idx=0):
        # print(f"[{process_idx}] Act and train\nt: {self.t} | t_start: {self.t_start} | t_max: {self.t_max}")
        state_var = torch.from_numpy(state_var).to(self.device)

        self.past_rewards[self.t-1] = reward

        if self.t - self.t_start == self.t_max:
            self.update(state_var, len(reward), process_idx=process_idx)

        self.past_states[self.t] = state_var

        pout, vout, inner_state = self.shared_model.pi_and_v(state_var)
        n, num_actions, h, w = pout.shape

        p_trans = pout.permute([0, 2, 3, 1])
        dist = Categorical(p_trans)
        action = dist.sample()
        log_p = torch.log(torch.clamp(p_trans, min=1e-9, max=1-1e-9))
        log_action_prob = torch.gather(log_p, 1, Variable(action.unsqueeze(-1))).view(n, 1, h, w)
        entropy = -torch.sum(p_trans * log_p, dim=-1).view(n, 1, h, w)

        # self.past_action_log_prob[self.t] = dist.log_prob(action).unsqueeze(dim=1).to(self.device)
        # self.past_action_entropy[self.t] = dist.entropy().unsqueeze(dim=1).to(self.device)
        self.past_action_log_prob[self.t] = log_action_prob
        self.past_action_entropy[self.t] = entropy
        self.past_values[self.t] = vout

        self.t += 1

        return action.detach().cpu().numpy(), inner_state.detach().cpu(), torch.exp(log_action_prob).detach().cpu()


    def stop_episode_and_train(self, state_var, reward, done=False, process_idx=0):
        print(f'[{process_idx}] Stop and Train')

        self.past_rewards[self.t-1] =reward
        if done:
            self.update(None, len(reward), process_idx=process_idx)
        else:
            self.update(state_var, len(reward), process_idx=process_idx)

    def get_statistics(self):
        return {
            "average_value": self.average_value,
            "average_entropy": self.average_entropy
        }