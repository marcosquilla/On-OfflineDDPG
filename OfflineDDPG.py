import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Tanh
from pytorch_sac_ae.video import VideoRecorder
import random
import dmc2gym
from tqdm import tqdm
import os

import warnings
warnings.filterwarnings("ignore")
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parameters
directory = 11 # Where to save whatever is saved and read the data from (there should be a data folder inside of ./policies/directory)
batch_size = 100
num_episodes = 500000
learning_ratePi = 1e-05
learning_rateQ = 1e-04
gamma = 0.999
tau = 0.01
Pi_structure = [128, 128, 128]
Q_structure = [128, 128, 128]
action_skip = 8
balance_dataset = False # Set to True to balance the dataset

# Initilise the environment
env = dmc2gym.make(
        domain_name='cartpole',
        task_name='balance',
        seed=1,
        visualize_reward=False,
        from_pixels=False,
        height=84,
        width=84,
        frame_skip=action_skip)

# Save the parameters used in a txt

with open('./policies/' + str(directory) + '/train_params.txt', 'w') as f:
	f.write('batch_size = ' + str(batch_size) + '\n')
	f.write('num_episodes = ' + str(num_episodes) + '\n')
	f.write('learning_ratePi = ' + str(learning_ratePi) + '\n')
	f.write('learning_rateQ = ' + str(learning_rateQ) + '\n')
	f.write('gamma = ' + str(gamma) + '\n')
	f.write('tau = ' + str(tau) + '\n')
	f.write('Pi_structure = ' + str(Pi_structure) + '\n')
	f.write('Q_structure = ' + str(Q_structure) + '\n')
	f.write('action_skip = ' + str(action_skip) + '\n')

# Load the data into the replay buffer. Balance the replay buffer if needed
replay = torch.load('./policies/' + str(directory) + '/data/' + os.listdir('./policies/' + str(directory) + '/data')[0])
for f in os.listdir('./policies/' + str(directory) + '/data')[1:]:
	replay = replay + torch.load('./policies/' + str(directory) + '/data/' + f)

if balance_dataset:
	rreplay = np.array(replay)
	final = []
	for i in rreplay:
		a = np.hstack((i,str(round(i[2],1)))) # Pick number of decimals to round to
		final.append(a)
	df = pd.DataFrame(final)
	df2 = df.groupby([5]).apply(lambda grp: grp.sample(n=1100)) # 1100 should be increased until the algorithm returns an error. It represents how many records of each class are taken
	replay = df2.iloc[:,0:5].to_numpy().tolist()

# Set up of both policy and Q networks. They are set up in a generic way so that it is easy to change parameters
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

class Q(nn.Module):
	def __init__(self, hidden_layers=Q_structure):
		super(Q, self).__init__()

		self.Lobs = Linear(5, int(hidden_layers[0]/2))
		self.Lact = Linear(1, int(hidden_layers[0]/2))
		self.hidden_in_features = hidden_layers[0]
        
		self.layers_hidden = []
		for neurons in hidden_layers[1:]:
			self.layers_hidden.append(Linear(self.hidden_in_features, neurons))
			self.hidden_in_features = neurons
			self.layers_hidden.append(ReLU())

		self.layers_hidden.append(Linear(hidden_layers[-1], 1))

		self.layers_hidden = nn.Sequential(*self.layers_hidden)
    
	def forward(self, obs, act):
		xobs = F.relu(self.Lobs(obs))
		xact = F.relu(self.Lact(act))
		x = torch.cat((xobs, xact), 1)
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
	sample = np.array(random.sample(replay, batch_size))
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

n_evals = 100 # Number of evals (points in graph)

reward_history = []

for i_episode in tqdm(range(num_episodes)):
	# Perform one step of the optimization
	_, _ = optimize_model()
	
    # Update the target network with Polyak avergaing
	target_net.update_params(policy_net.state_dict(), tau)
	Q_target.update_params(Q_policy.state_dict(), tau)
	
	# Eval (with video), plot and save weights and biases from the policy.
	if i_episode % int(num_episodes/n_evals) == 0:
		with torch.no_grad():
			er = []
			for _ in range(10): # 10 evals are done and then the average is plotted
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
				er.append(total_reward)
		# Save latest policy and the video
		torch.save(policy_net.state_dict(), './policies/' + str(directory) + '/last_policy_params')
		vidname = './policies/'+ str(directory) + '/last.mp4'
		video.save(vidname)
		
		# Logging of losses, estimated return and return from eval
		reward_history.append([i_episode, np.array(er).mean(), np.array(er).std()])
		areward_history = np.array(reward_history)
		
		# Save this policy and eval video if they have the highest return so far
		if areward_history[-1,1] == max(areward_history[:,1]):
			torch.save(policy_net.state_dict(), './policies/' + str(directory) + '/best_policy_params')
			vidname = './policies/'+ str(directory) + '/best.mp4'
			video.save(vidname)
			
		# Plotting
		plt.figure(figsize=(15,10))
		plt.plot(areward_history[:,0], areward_history[:,1], label='Mean reward')
		plt.plot(areward_history[:,0], areward_history[:,2], label='Std reward')
		plt.title('Environment reward')
		plt.legend(loc='upper left')
		# Save the figure and the values for plotting later
		if balance_dataset:
			plt.savefig('./policies/' + str(directory) + '/Reward plotb.png')
			np.save('./policies/' + str(directory) + '/rewardsb', areward_history)
		else:
			plt.savefig('./policies/' + str(directory) + '/Reward plot.png')
			np.save('./policies/' + str(directory) + '/rewards', areward_history)
