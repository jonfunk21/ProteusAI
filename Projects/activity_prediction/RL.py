import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
sys.path.append('../../src/')
sys.path.append('../activity_prediction/')
import proteusAI.ml_tools.esm_tools.esm_tools as esm_tools
import random
from activity_predictor import FFNN
import os
import numpy as np

alphabet = esm_tools.alphabet.to_dict()

class Environment:
    def __init__(self, initial_seq, mut_depth=3, max_len=1024, aa_num=20, batch_size=10, mutation_penalty=-10,
                 models_path='../activity_prediction/checkpoints/'):
        self.initial_seq = initial_seq
        self.mut_depth = mut_depth
        self.counter = 0
        self.max_len = max_len
        self.aa_num = aa_num
        self.batch_size = batch_size
        self.mutation_penalty = mutation_penalty

        self.AAs = ('A', 'C', 'D', 'E', 'F', 'G', 'H',
                    'I', 'K', 'L', 'M', 'N', 'P', 'Q',
                    'R', 'S', 'T', 'V', 'W', 'Y')
        self.seqs = [self.initial_seq] * self.batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.ensemble = []
        self.models_path = models_path
        for m in os.listdir(models_path):
            if 'model' in m:
                model = FFNN(input_size=1280, output_size=1, hidden_layers=[1280, 1280], dropout_rate=0.2)
                model.load_state_dict(torch.load(os.path.join(models_path, m)))
                model.to(self.device)
                model.eval()
                self.ensemble.append(model)

        self.s0 = self.compute_state(self.seqs)
        # Initialize the environment
        self.reset()

    def reset(self):
        self.counter = 0
        return self.s0

    def step(self, a):
        # Process the actions vector
        self.seqs, penalties = self.mutate(a)

        # Compute sequence representations (states)
        seq_reps = self.compute_state(self.seqs)

        # Compute activity values as rewards
        rewards = self.dummy_activity_value(seq_reps) - penalties

        # break if max mut depth has been reached.
        self.counter += 1
        done = False
        if self.counter == self.mut_depth:
            done = True

        return seq_reps, rewards, done, {}  # {} could be information

    def mutate(self, a):
        seqs = self.seqs
        # You should provide the implementation of this method
        penalties = torch.zeros(len(seqs))
        for i, seq in enumerate(seqs):
            max_len = len(seq)
            pos, mut = self.get_action(a[i])

            # penalize predictions outside the sequence length or missense mutations
            if pos > max_len:
                penalties[i] = self.mutation_penalty
            elif mut == seq[pos]:
                penalties[i] = self.mutation_penalty
            else:
                mut_seq = seq[:pos] + mut + seq[pos + 1:]
                seqs[i] = mut_seq

        self.seqs = seqs
        return seqs, penalties

    def compute_state(self, seqs):
        results, batch_lens, batch_labels, _ = esm_tools.esm_compute(seqs)
        r = esm_tools.get_seq_rep(results, batch_lens)
        return torch.stack(r)

    def dummy_activity_value(self, seq_reps):
        activities = []
        for model in self.ensemble:
            act = model(seq_reps.to(self.device))
            activities.append(act.detach().cpu().numpy())

        average_activity = np.mean(activities, axis=0)
        return torch.from_numpy(average_activity)

    def get_action(self, a):
        # calculate position
        pos = a // len(self.AAs)

        # calculate mutation index
        mut_index = a % len(self.AAs)

        # get the corresponding residue
        mut = self.AAs[mut_index]
        return pos, mut


class QNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, dropout_rate):
        super(QNet, self).__init__()
        self.output_size = output_size
        self.layers = []
        self.layers.append(nn.Linear(input_size, hidden_layers[0]))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(dropout_rate))

        for i in range(1, len(hidden_layers)):
            self.layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout_rate))

        self.layers.append(nn.Linear(hidden_layers[-1], output_size))
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)

    def epsilongreedy(self, state, epsilon):
        if random.random() < epsilon:
            return torch.randint(low=0, high=self.output_size, size=(state.size(0),))
        else:
            with torch.no_grad():
                return torch.argmax(self(state), dim=1)


def train_qlearn(environment, Qnet, alpha=0.001, gamma=0.9, epsilon=0.05, max_epochs=10000):
    optimizer = torch.optim.Adam(Qnet.parameters(), lr=alpha)
    max_reward = float('-inf')
    rewards = []  # list to store total rewards for each epoch

    # wrap your range with tqdm for a progress bar
    for epoch in tqdm(range(max_epochs)):
        s = environment.reset()

        total_reward = 0
        while True:
            a = Qnet.epsilongreedy(s, epsilon)
            s_next, r, done, _ = environment.step(a)

            target = r + gamma * torch.max(Qnet(s_next))
            output = Qnet(s)[torch.arange(s.size(0)), a]

            total_reward += r.sum().item()

            loss = (target - output).pow(2).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if done:
                break
            else:
                s = torch.Tensor(s_next)

        # Save the model if it has the highest total reward so far
        if total_reward > max_reward:
            max_reward = total_reward
            torch.save(Qnet.state_dict(), 'checkpoints/RL_agent.pth')

        rewards.append(total_reward)  # store the total reward

        # Plot total reward every 100 epochs
        if epoch % 100 == 0 and epoch > 0:
            plt.figure(figsize=(12, 6))
            plt.plot(rewards)
            plt.xlabel('Epoch')
            plt.ylabel('Total Reward')
            plt.savefig('checkpoints/RL_agent_rewards.png')
            break

    return Qnet


seq = "MAPTLSEQTRQLVRASVPALQKHSVAISATMCRLLFERYPETRSLCELPERQIHKIASALLAYARSIDNPSALQAAIRRMVLSHARAGVQAVHYPLYWECLRDAIKEVLGPDATETLLQAWKEAYDFLAHLLSTKEAQVYAVLAE"
model_dim = 1280
max_seq_len = 1024
num_residues = 20
action_space = max_seq_len * num_residues

sum_sq = 0
environment = Environment(seq, batch_size=10)
Qnet = QNet(model_dim, action_space, [model_dim, model_dim], 0.2)

train_qlearn(environment, Qnet)