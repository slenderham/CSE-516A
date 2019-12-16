import numpy as np
# from scipy.special import logsumexp as logsumexp
from matplotlib import pyplot as plt

SMALL = np.finfo(float).eps;

class Agent:
	def __init__(self):
		self.num_actions = 2;
		self.costs = np.zeros(self.num_actions);
		self.freq = np.zeros(self.num_actions);
		self.pseudocount = np.zeros(self.num_actions);

	def update_own_state(self, choice, cost):
		self.freq[choice] += 1;
		self.costs[choice] += cost;

	def update_global_state(self, counts, costs):
		self.freq += counts;
		self.costs += costs*counts;

	def greedy(self):
		Qvals = self.costs/(self.freq+SMALL);
		return Qvals, self.break_tie(Qvals);

	def eps_greedy(self, eps):
		# prob of epsilon of giving all actions infinity cost, so that none will standout
		greedy = np.random.binomial(1, 1-eps, 1);

		if (greedy):
			Qvals = self.costs/(self.freq+SMALL);
			return Qvals, self.break_tie(Qvals);
		else:
			Qvals = np.zeros_like(self.costs);
			return Qvals, np.random.randint(self.num_actions);

	def UCB1(self, sigma):

		# rewards is always positive, suppose the negative cost follows log normal distribution
		Qvals = (self.costs)/(self.freq+SMALL) - np.sqrt( sigma*np.log(np.sum(self.freq)+SMALL) / (self.freq+SMALL) );
		return Qvals, self.break_tie(Qvals);

	def dirichlet(self, highways, num_agents, count_so_far):
		concentration = np.ones(self.num_actions);
		mode = (self.freq+concentration+count_so_far-1)/(np.sum(self.freq+concentration+count_so_far)-self.num_actions)*num_agents;

		Qvals = np.zeros_like(mode);

		for idx, h in enumerate(highways):
			Qvals[idx] = h.cost(mode[idx]);

		return Qvals, self.prob_matching(Qvals);

	# def prob_matching(self, x):
	# 	return np.random.choice(np.arange(x.shape[0]), 1, p = np.exp(-10*x-logsumexp(-10*x)));

	def break_tie(self, x):
		return np.random.choice(np.flatnonzero(x == x.min()));

class highway:
	a = 0
	b = 0

	# cost = ax+b
	def __init__(self, a, b):
		self.a = a
		self.b = b

	def cost(self, t):
		return self.a*t+self.b

agent_num=100;
num_its = 10;
play_times = 300;

h1 = highway(0.4, 0);
h2 = highway(0.4, 0);
h3 = highway(0, 45);
h4 = highway(0, 45);

highways = [h1, h2, h3, h4]
options = [
# 			[0, 1],
		   	[0, 2],
		   	[3, 1],
# 		   	 [3, 2],
		   ]

eps_to_try = [1, 0.9, 0.5, 0.1, 0.05, 0.01, 0.001]
num_agens_to_try = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500];
costs_hist = np.zeros((num_its, len(eps_to_try)));
costs_hist_decay = np.zeros((num_its, len(eps_to_try)));

def run(agent_num, param, play_times):

	agents = [];

	for i in range(agent_num):
		agents.append(Agent());

	for i in range(play_times):
		everyone_total_costs = 0;
		choices = np.zeros(agent_num, dtype=int);
		counts = np.zeros(len(highways));

		for idx, a in enumerate(agents):
			if (i==0):
				action = np.random.randint(len(options));
			else:
				# Qvals, action = a.dirichlet(highways, agent_num, np.zeros(3));
# 				Qvals, action = a.eps_greedy(min(1, 4/(i)));
				Qvals, action = a.UCB1(2);

			choices[idx] = action;
			for k in options[action]:
				counts[k] += 1;

		costs = np.zeros(len(highways));
		for idx, h in enumerate(highways):
			costs[idx] = h.cost(counts[idx]);

		for idx, a in enumerate(agents):
			own_total_cost = 0;
			for k in options[choices[idx]]:
				own_total_cost += costs[k];
			everyone_total_costs += own_total_cost;
			a.update_own_state(choices[idx], own_total_cost);
			# a.update_global_state(counts, costs);

	return counts, costs, everyone_total_costs;


# total_costs = [];
# for num in num_agens_to_try:
# 	total_costs.append([]);
# 	for i in range(10):
# 		counts, costs, everyone_total_costs = run(num, 0.9, play_times);
# 		total_costs[-1].append(everyone_total_costs/num);
# 	print(total_costs[-1]);

# print(total_costs);


# highways = [h1, h4];

total_costs_no_high = [];
for i in range(10):
 	counts, costs, everyone_total_costs = run(100, 0.9, 500);
 	total_costs_no_high.append(everyone_total_costs/100);
print(total_costs_no_high);



# final_choices_no_decay = []
# final_choices_decay = []


# for xx in range(num_its):
# 	print(xx, ": no decay");
# 	for edx, eps in enumerate(eps_to_try):
# 		agents = [];

# 		for i in range(agent_num):
# 			agents.append(Agent());
# 		for i in range(play_times):
# 			everyone_total_costs = 0;
# 			choices = np.zeros(agent_num, dtype=int);
# 			counts = np.zeros(len(options));

# 			for idx, a in enumerate(agents):
# 				if (i==0):
# 					action = np.random.randint(len(options));
# 				else:
# 					# Qvals, action = a.dirichlet(highways, agent_num, np.zeros(3));
# 					Qvals, action = a.eps_greedy(eps);


# 				choices[idx] = action;
# 				for k in options[action]:
# 					counts[k] += 1;

# 			costs = np.zeros(len(highways));
# 			for idx, h in enumerate(highways):
# 				costs[idx] = h.cost(counts[idx]);
# 			for idx, a in enumerate(agents):
# 				own_total_cost = 0;
# 				for k in options[choices[idx]]:
# 					own_total_cost += costs[k];
# 				everyone_total_costs += own_total_cost;
# 				a.update_own_state(choices[idx], own_total_cost);

# 		print(counts);
# 		final_choices_no_decay.append(counts);

# 		costs_hist[xx, edx] = everyone_total_costs/agent_num;

# 	print("=====================================");
# 	print(xx, ": decay");


# 	for edx, eps in enumerate(eps_to_try):
# 		agents = [];

# 		for i in range(agent_num):
# 			agents.append(Agent());
# 		for i in range(play_times):
# 			everyone_total_costs = 0;
# 			choices = np.zeros(agent_num, dtype=int);
# 			counts = np.zeros(len(highways));

# 			for idx, a in enumerate(agents):
# 				if (i==0):
# 					action = np.random.randint(len(highways));
# 				else:
# 					# Qvals, action = a.dirichlet(highways, agent_num, np.zeros(3));
# 					Qvals, action = a.eps_greedy(min(1, eps/(i/4)));


# 				choices[idx] = action;
# 				for k in options[action]:
# 					counts[k] += 1;

# 			costs = np.zeros(len(highways));
# 			for idx, h in enumerate(highways):
# 				costs[idx] = h.cost(counts[idx]);
# 			for idx, a in enumerate(agents):
# 				own_total_cost = 0;
# 				for k in options[choices[idx]]:
# 					own_total_cost += costs[k];
# 				everyone_total_costs += own_total_cost;
# 				a.update_own_state(choices[idx], own_total_cost);
# 				# a.update_global_state(counts, costs);

# 		print(counts);
# 		final_choices_decay.append(counts);
# 		costs_hist_decay[xx, edx] = everyone_total_costs/agent_num;

# 	print("=====================================");

# fig, (ax1, ax2) = plt.subplots(1, 2);
# ax1.violinplot(costs_hist);
# ax2.violinplot(costs_hist_decay);
# ax1.plot(np.arange(1, 8), np.mean(np.array(costs_hist), axis=0), 'bo', markersize=3);
# ax2.plot(np.arange(1, 8), np.mean(np.array(costs_hist_decay), axis=0), 'bo', markersize=3);
# ax1.set_xlabel(r"$\epsilon$");
# ax2.set_xlabel(r"$\epsilon$");
# ax1.set_ylabel("average cost");
# ax1.set_title( r"$\epsilon$-greedy")
# ax2.set_title(r"decaying $\epsilon$-greedy")
# ax1.set_xticks(np.arange(1, 8))
# ax2.set_xticks(np.arange(1, 8))
# ax1.set_xticklabels(eps_to_try)
# ax2.set_xticklabels(eps_to_try)
# plt.show();


fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.violinplot(np.array(total_costs_eps_greedy).T, showmeans=False)
ax2.violinplot(np.array(total_costs_UCB1).T, showmeans=False)
ax3.violinplot(np.array(total_costs_coord).T, showmeans=False)
ax1.plot(np.arange(1, 11), np.mean(np.array(total_costs_eps_greedy), axis=1), 'bo', markersize=3);
ax2.plot(np.arange(1, 11), np.mean(np.array(total_costs_UCB1), axis=1), 'bo', markersize=3);
ax3.plot(np.arange(1, 11), np.mean(np.array(total_costs_coord), axis=1), 'bo', markersize=3);
ax1.set_xticks(np.arange(1, 11))
ax2.set_xticks(np.arange(1, 11))
ax3.set_xticks(np.arange(1, 11))
ax1.set_xticklabels([50, 100, 150, 200, 250, 300, 350, 400, 450, 500], rotation=45)
ax2.set_xticklabels([50, 100, 150, 200, 250, 300, 350, 400, 450, 500], rotation=45)
ax3.set_xticklabels([50, 100, 150, 200, 250, 300, 350, 400, 450, 500], rotation=45)
ax1.set_title(r"$\epsilon$-greedy")
ax2.set_title("UCB1")
ax3.set_title("Coordinated Actions")
ax1.set_ylabel("Mean Cost per Agent")
ax1.set_xlabel("Number of Agents")
ax2.set_xlabel("Number of Agents")
ax3.set_xlabel("Number of Agents")

plt.violinplot([total_costs_no_high_eps,total_costs_no_high_UCB1, total_costs_no_high_coord], positions=[1, 2, 3]);
plt.violinplot([total_costs_eps_greedy[1], total_costs_UCB1[1], total_costs_coord[1]], positions=[4,5,6]);
plt.xticks(range(1, 7), labels=[r"$\epsilon$-greedy, no highway", "UCB1, no highway", "Coordinated, no highway", r"$\epsilon$-greedy, highway", "UCB1, highway", "Coordinated, highway"], rotation=45)
plt.ylabel("Mean Cost Per Agent")

labels=r"$\epsilon$-greedy, no highway"
labels="UCB1, no highway"
labels="Coordinated Actions, no highway"