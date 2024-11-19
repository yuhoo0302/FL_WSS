import random
import torch.nn as nn
import numpy as np
from torchvision.models import resnet18
from torch.autograd import Variable
import torch
import ipdb


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.action_space = args.action_space
        self.conv0 = nn.Conv2d(6, 3, kernel_size=1, stride=1)
        self.feature = resnet18(pretrained=True) # replace to a 3d resnet for 3d agent
        self.linear1 = nn.Linear(1000, 512)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(0.5)

        self.fc0_adv = nn.Linear(in_features=512, out_features=self.action_space)
        self.fc0_val = nn.Linear(in_features=512, out_features=1)

        self.fc1_adv = nn.Linear(in_features=512, out_features=self.action_space)
        self.fc1_val = nn.Linear(in_features=512, out_features=1)

    def forward(self, x):

        x = self.feature(self.conv0(x.float()))
        x = x.view(x.size(0), -1)
        x = self.drop(self.relu(self.linear1(x)))

        adv0 = self.fc0_adv(x)
        val0 = self.fc0_val(x).expand(x.size(0), self.action_space)
        adv1 = self.fc1_adv(x)
        val1 = self.fc1_val(x).expand(x.size(0), self.action_space)

        a0 = torch.squeeze(val0 + adv0 - adv0.mean(1).unsqueeze(1).expand(x.size(0), self.action_space))
        a1 = torch.squeeze(val1 + adv1 - adv1.mean(1).unsqueeze(1).expand(x.size(0), self.action_space))

        if len(torch.stack([a0, a1], dim=0).size()) == 2:
            x = torch.stack([a0, a1], dim=0).unsqueeze(1)
        else:
            x = torch.stack([a0, a1], dim=0)
        x = x.permute(1, 0, 2)
        return x


class Agent(object):

    def __init__(self, args):
        self.eval_net = Net(args).cuda()
        self.target_net = Net(args).cuda()
        self.batch_size = args.batch_size
        self.target_step_counter = args.target_step_counter
        self.replay_buffer = NaivePrioritizedBuffer(args.memory_capacity)
        self.optimizer = torch.optim.AdamW(self.eval_net.parameters(),lr=args.lr,weight_decay=args.weight_decay)
        self.action_space = args.action_space
        self.loss_func = nn.MSELoss()
        self.learning_step = 0
        self.epsilon = args.epsilon
        self.gamma = args.gamma
        self.ep_frequency = args.ep_frequency
        self.frame_idx = 0

    def select_action(self, x, return_q=False, random_action=True):
        self.eval_net.eval()
        x = torch.from_numpy(x[np.newaxis, :, :, :]).cuda()
        if random_action:
            if random.random() < self.epsilon:
                action = self.eval_net.forward(x)
                action_value_arr = action.data.cpu().numpy()
            else:
                action_value_arr = [np.array([random.random() for _ in range(self.action_space)], dtype=np.float32) for _ in range(2)]
        else:
            action = self.eval_net.forward(x)
            action_value_arr = action.data.cpu().numpy()

        action_value_arr = np.squeeze(action_value_arr)
        action = action_value_arr.argmax(1)
        if return_q:
            return action, action_value_arr
        else:
            return action

    def learn(self, beta):
        self.eval_net.train()
        self.target_net.eval()
        if self.learning_step % self.target_step_counter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learning_step += 1
        if len(self.replay_buffer) < self.batch_size:
            return 0
        if self.learning_step % self.ep_frequency == 0 and self.epsilon < 0.95:
            self.epsilon *= 1.01
        state, action, reward, next_state, done, indices, weights = self.replay_buffer.sample(self.batch_size, beta)

        state = Variable(torch.from_numpy(state)).cuda()
        next_state = Variable(torch.from_numpy(next_state)).cuda()
        reward = Variable(torch.from_numpy(reward)).cuda()
        action = Variable(torch.from_numpy(action).unsqueeze(2)).cuda()
        weights = Variable(torch.from_numpy(weights)).cuda()
        q_eval = self.eval_net(state)
        q_eval = q_eval.gather(2, action).squeeze()
        q_next_t = self.target_net(next_state).detach()
        q_next_e = self.eval_net(next_state).detach()
        q_next = q_next_t.gather(2, q_next_e.max(2)[1].unsqueeze(1).permute(0,2,1)).squeeze()
        q_target = (q_next * self.gamma) + reward
        loss = self.loss_func(q_eval, q_target) * weights
        prios = loss + 1e-5
        loss = loss.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.replay_buffer.update_priorities(indices, prios.data.cpu().numpy())
        self.optimizer.step()

        return loss.item()


class NaivePrioritizedBuffer(object):
    def __init__(self, capacity, prob_alpha=0.6):
        self.prob_alpha = prob_alpha
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):

        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        
        probs = prios ** self.prob_alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        batch = zip(*samples)
        batch = [data for data in batch]

        states = np.concatenate(batch[0], axis=0)
        actions = np.array(batch[1], dtype=np.int64)
        rewards = np.array(batch[2], dtype=np.float32)
        next_states = np.concatenate(batch[3], axis=0)
        dones = batch[4]

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)


def beta_by_frame(frame_idx):
    beta_start = 0.4
    beta_frames = 2000
    return min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)
