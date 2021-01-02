import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Tanh
from pytorch_sac_ae.utils import ReplayBuffer
from pytorch_sac_ae.video import VideoRecorder
import dmc2gym
from tqdm import tqdm
import os

import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parameters
directory = 111 # Save location of the dataset, plots, video and policy
batch_size = 100
num_episodes = 10000
learning_ratePi = 0.0001
learning_rateQ = 0.001
gamma = 0.999
tau = 0.01
Pi_structure = [128, 128, 128]
Q_structure = [128, 128, 128]
action_skip = 8
diversity = True # Set to False to not include random actions and use the ordinary algorithm

# Initalise the environment
env = dmc2gym.make(
        domain_name='cartpole',
        task_name='balance',
        seed=1,
        visualize_reward=False,
        from_pixels=False,
        height=84,
        width=84,
        frame_skip=action_skip)

# Create directory for saving policies, videos, plot. Then save the parameters used in a txt
try:
    os.mkdir('./policies/' + str(directory))
except:
    pass

with open('./policies/' + str(directory) + '/train_params.txt', 'w') as f:
	f.write('batch_size = ' + str(batch_size) + '\n')
	f.write('num_episodes = ' + str(num_episodes) + '\n')
	f.write('learning_ratePi = ' + str(learning_ratePi) + '\n')
	f.write('learning_rateQ = ' + str(learning_rateQ) + '\n')
	f.write('gamma = ' + str(gamma) + '\n')
	f.write('tau = ' + str(tau) + '\n')
	f.write('Pi_structure = ' + str(Pi_structure) + '\n')
	f.write('Q_structure = ' + str(Q_structure) + '\n')
	f.write('aciton_skip = ' + str(action_skip) + '\n')

# Initialise the replay memory
class ReplayMemory(object):

	def __init__(self, capacity):
		self.capacity = capacity
		self.memory = []
		self.position = 0

	def push(self, new_row):
		# Saves a transition
		if len(self.memory) < self.capacity:
			self.memory.append(new_row)

	def sample(self, batch_size):
		return random.sample(self.memory, batch_size)

	def __len__(self):
		return len(self.memory)
replay = ReplayMemory(10000000)

# Set up of both policy and Q networks
class Pi(nn.Module):
	def __init__(self, hidden_layers=Pi_structure):
		super(Pi, self).__init__()
        
		self.hidden_in_features = 5
        
		self.layers_hidden = []
		for neurons in hidden_layers:
			self.layers_hidden.append(Linear(self.hidden_in_features, neurons))
			self.hidden_in_features = neurons
			self.layers_hidden.append(ReLU())

		self.layers_hidden.append(Linear(hidden_layers[-1], 1))
		self.layers_hidden.append(Tanh()) # This function is necessary as the action space ranges from -1 to 1
		self.layers_hidden = nn.Sequential(*self.layers_hidden)
    
	def forward(self, x):
		return self.layers_hidden(x)
        
	def update_params(self, new_params, tau):	# Polyak averaging to update the target network
		params = self.state_dict()
		for k in params.keys():
			params[k] = (1-tau) * params[k] + tau * new_params[k]
		self.load_state_dict(params)

# Set up of both policy and Q networks. They are set up in a generic way so that it is easy to change parameters
class Q(nn.Module):
	def __init__(self, hidden_layers=Q_structure):
		super(Q, self).__init__()
        
		self.ObsSize = 5
		self.ActionSize = 1
		self.Lobs = Linear(self.ObsSize, int(hidden_layers[0]/2))
		self.Lact = Linear(self.ActionSize, int(hidden_layers[0]/2))
		self.hidden_in_features = hidden_layers[0]
        
		self.layers_hidden = []
		for neurons in hidden_layers[1:]:
			self.layers_hidden.append(Linear(self.hidden_in_features, neurons))
			self.hidden_in_features = neurons
			self.layers_hidden.append(ReLU())

		self.layers_hidden.append(Linear(hidden_layers[-1], 1))

		self.layers_hidden = nn.Sequential(*self.layers_hidden)
    
	def forward(self, obs, act):
		xobs = self.Lobs(obs)
		xact = self.Lact(act)
		x = F.relu(torch.cat((xobs, xact), 1))
		return self.layers_hidden(x)
        
	def update_params(self, new_params, tau):	# Polyak averaging to update the target network
		params = self.state_dict()
		for k in params.keys():
			params[k] = (1-tau) * params[k] + tau * new_params[k]
		self.load_state_dict(params)

# Initialisation of both policy and target networks, as well as their optimisers
policy_net = Pi().to(device)
target_net = Pi().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

Q_policy = Q().to(device)
Q_target = Q().to(device)
Q_target.load_state_dict(Q_policy.state_dict())
Q_target.eval()

Pi_optimizer = optim.Adam(policy_net.parameters(), lr=learning_ratePi)
Q_optimizer = optim.Adam(Q_policy.parameters(), lr=learning_rateQ)

# Define the function that will train a batch
def optimize_model():
	# Random sample from the replay buffer and split of the different components from the sample
	sample = np.array(replay.sample(batch_size))
	obses = torch.Tensor(np.vstack(sample[:,0])).to(device)
	actions = torch.Tensor(np.vstack(sample[:,1])).to(device)
	rewards = torch.Tensor(np.vstack(sample[:,2])).to(device)
	next_obses = torch.Tensor(np.vstack(sample[:,3])).to(device)
	not_dones = torch.Tensor(np.vstack(sample[:,4])).to(device)
	
	q_target = Q_target(next_obses, target_net(next_obses)) # Use the Q network to estimate the rewards from the target network's actions    
	y = q_target * gamma * not_dones + rewards # Compute the estimated return, by adding the return for the next action and the reward for the current action
	
	# Optimize both networks
    # Compute mean-squared Bellman error for the Q network(MSBE)
	Qloss = F.mse_loss(Q_policy(obses, actions), y)
	Q_optimizer.zero_grad()
	Qloss.backward()
	Q_optimizer.step()

	# The loss for the policy is just the negative value of the Q function. By doing this we look for the actions that maximise the return
	Piloss = -Q_policy(obses, policy_net(obses)).mean()	
	Pi_optimizer.zero_grad()
	Piloss.backward()
	Pi_optimizer.step()
		
	return -Piloss, y.mean() # Return these values for plotting

n_evals = 100 # Number of evals
n_saves = 10 # Number of files the data will be split

reward_history = []
er = []

for i_episode in tqdm(range(num_episodes)):
	# Test the policy online and save the replay on the replay memory
	with torch.no_grad():
		if diversity:
			random_actions = True if random.random() < (i_episode/num_episodes)**1/2 else False # 0 random actions at the start of training to 50% at the end. Linear increase
		else:
			random_actions = False
		done = False
		obs = torch.from_numpy(env.reset())
		total_reward = 0
		while not done:
			if not random_actions:
				noise = 0 if random.random() > 0.05 + (0.95 - 0.05) * (1 - i_episode/num_episodes)**2 else (torch.rand(1)*2-1)*1e-1 # Included noise for exploration	
				action = policy_net(obs.type(torch.FloatTensor).to(device)).cpu() + noise
			else:
				action = torch.rand(1)*2-1
			action[action > 1] = 1
			action[action < -1] = -1
			next_obs, reward, done, _ = env.step(action.detach().numpy())
			replay.push((obs.cpu().detach().numpy(), action.detach().numpy(), reward, next_obs, not done)) # Save the current step in the replay buffer
			obs = torch.from_numpy(next_obs)
			total_reward += reward
		er.append(total_reward)

	# Perform 20 steps of the optimization
	for _ in range(20):
		_, _ = optimize_model()
	
	# Update the target network with Polyak avergaing
	target_net.update_params(policy_net.state_dict(), tau)
	Q_target.update_params(Q_policy.state_dict(), tau)

	# Eval (with video) and save weights and biases from the policy.
	if i_episode % int(num_episodes/n_evals) == 0:
		reward_history.append([i_episode, np.array(er).mean(), np.array(er).std()])
		areward_history = np.array(reward_history)
		
		plt.figure(figsize=(15,10))
		plt.plot(areward_history[:,0], areward_history[:,1], label='Mean reward')
		plt.plot(areward_history[:,0], areward_history[:,2], label='Std reward')
		plt.legend(loc='upper left')
		plt.title('Environment reward')
		plt.savefig('./policies/' + str(directory) + '/Reward plot.png')
		np.save('./policies/' + str(directory) + '/rewards', areward_history)
		er = []
		
		with torch.no_grad():
			video = VideoRecorder('./')
			obs = env.reset()
			video.init(enabled=True)
			done = False
			total_reward = 0
			while not done:
				action = policy_net(torch.Tensor(obs)[None, ...].to(device))
				obs, ereward, done, _ = env.step(action.cpu().detach().numpy()[0])
				video.record(env)
				total_reward += ereward
			vidname = './policies/'+ str(directory) + '/last.mp4'
			video.save(vidname)
			torch.save(policy_net.state_dict(), './policies/' + str(directory) + '/last_policy_params')

	if i_episode % int(num_episodes/n_saves) == 0: # Save a file
			torch.save(replay.memory[-int(num_episodes/n_saves*1000/action_skip):], './policies/' + str(directory) + '/data/replayCart' + str(i_episode))
