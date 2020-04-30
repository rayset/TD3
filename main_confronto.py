import numpy as np
#import torch
import gym
import argparse
import os
import scipy
import scipy.io


import utils
import TD3
import OurDDPG
import DDPG
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rewards_plot=[];
Q_plots=[]
d_rewards_plot=[]
time_plot=[]
states=[]
rewards=[]
# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy1, policy2, env_name, seed, eval_episodes=200):
	eval_env = gym.make(env_name)
	eval_env.seed(seed+100)

	avg_reward = 0.
	for _ in range(eval_episodes):
		discounted_reward=0
		state, done = eval_env.reset(), False
		states.append(np.zeros((1,5)))
		#eval Q error
		#target_Q1, target_Q2=policy.critic_target(torch.FloatTensor(state.reshape(1, -1)).to(device), policy.actor(torch.FloatTensor(state.reshape(1, -1)).to(device)))
		#minQ=torch.min(target_Q1, target_Q2).cpu().data.numpy().flatten()
		#print("---------------------------------------")
		#print(f"Min value of critic's Q(s(0),a(0)): {minQ}")
		#Q_plots.append(minQ)
		i=0
		while not done and i<=150:
			if state[4]>99:
			#	print('switched policy',state[4])
				policy=policy2
				state[4]=0
			else:
				policy=policy1
			action = policy.select_action(np.array(state))
			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward
			states.append(state)
			#print(reward)
			discounted_reward=discounted_reward+reward*0.95**(i-1)
			i=i+1

		print(f"surv time= {i}")
		print("---------------------------------------")
		rewards.append(reward)
		d_rewards_plot.append(discounted_reward)
		rewards_plot.append(reward)
		time_plot.append(i)
	avg_reward /= eval_episodes

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	scipy.io.savemat("TD3_confronto.mat",{'rewards':rewards_plot,'d_rewards':d_rewards_plot,'Q_values':Q_plots,'surv_time':time_plot,'states':states,'rewards':rewards})

	return avg_reward


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
	parser.add_argument("--env", default="HalfCheetah-v2")          # OpenAI gym environment name
	parser.add_argument("--seed", default=1992, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--start_timesteps", default=25e3, type=int)# Time steps initial random policy is used
	parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
	parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.95)                 # Discount factor
	parser.add_argument("--tau", default=0.01)                     # Target network update rate
	parser.add_argument("--policy_noise", default=0.3)              # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.6)                # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
	parser.add_argument("--save_model", default=1, action="store_true")        # Save model and optimizer parameters
	parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
	args = parser.parse_args()


	file_name = "TD3_baseline"
	print("---------------------------------------")
	print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	if not os.path.exists("./results"):
		os.makedirs("./results")

	if args.save_model and not os.path.exists("./models"):
		os.makedirs("./models")

	env = gym.make(args.env)

	# Set seeds
	env.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)

	state_dim = 5#env.observation_space.shape[0]
	action_dim = env.action_space.shape[0]
	max_action = float(env.action_space.high[0])

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
	}

	# Initialize policy
	if args.policy == "TD3":
		# Target policy smoothing is scaled wrt the action scale
		kwargs["policy_noise"] = args.policy_noise * max_action
		kwargs["noise_clip"] = args.noise_clip * max_action
		kwargs["policy_freq"] = args.policy_freq
		policy1 = TD3.TD3(**kwargs)

		kwargs["policy_noise"] = 0.4 * max_action
		kwargs["noise_clip"] = 0.8* max_action

		policy2 = TD3.TD3(**kwargs)

	elif args.policy == "OurDDPG":
		policy = OurDDPG.DDPG(**kwargs)
	elif args.policy == "DDPG":
		policy = DDPG.DDPG(**kwargs)

	policy_file = "TD3"
	policy2.load(f"./models/{policy_file}")
	policy_file = "TD3_2"
	policy1.load(f"./models/{policy_file}")

	evaluations = [eval_policy(policy1,policy2, args.env, args.seed)]
