import numpy as np

class Agent:
	def __init__(self):
		self.num_actions = 3;
		self.costs = np.zeros(self.num_actions);
		self.freq = np.ones(self.num_actions);

	def update_state(self, choice, cost):
		self.freq[choice] += 1;
		self.costs[choice] += cost;

	def greedy(self):
		Qvals = self.costs/self.freq;
		return Qvals, np.argmin(Qvals);

	def eps_greedy(self, eps):

		# prob of epsilon of giving all actions infinity cost, so that none will standout

		greedy = np.random.binomial(1, 1-eps, 1);

		if (greedy):
			Qvals = self.costs/self.freq;
			return Qvals, np.argmin(Qvals);
		else:
			Qvals = np.zeros_like(self.costs);
			return Qvals, np.random.randint(self.num_actions);
		

	def UCB1(self):

		# rewards is always negative, suppose the negative cost follows log normal distribution

		Qvals = -(np.log(-self.costs)/self.freq - np.sqrt(2*np.log(np.sum(self.freq)))/self.freq);
		return Qvals, np.argmin(Qvals);



class highway:
    a = 0
    b = 0

    # cost = ax+b
    def __init__(self, a, b):
        self.a = a
        self.b = b
    
    def cost(self, t):
        return self.a*t+self.b


h1 = highway(2,0)
h2 = highway(3,0)
h3 = highway(0,200)
h4 = highway(0,1000)
highways = [h1, h2, h3, h4]
agents = [];

agent_num=100;
play_times = 1000;

for i in range(agent_num):
	agents.append(Agent());


for i in range(play_times):
	choices = np.zeros(agent_num, dtype=int)-1;
	counts = np.zeros(len(highways));
	for idx, a in enumerate(agents):
		if (i==0):
			action = np.random.randint(len(highways)-1);
		else:
			Qvals, action = a.UCB1();

		choices[idx] = action;
		counts[action] += 1;

	costs = np.zeros(len(highways));
	for idx, h in enumerate(highways):
		costs[idx] = h.cost(counts[idx]);

	for idx, a in enumerate(agents):
		a.update_state(choices[idx], costs[choices[idx]]);

	print(counts);
