import random
from multiarm import Agent
# Definitions
# Highway (Linear functions)
class highway:
    a = 0
    b = 0

    # cost = ax+b
    def __init__(self, a, b):
        self.a = a
        self.b = b
    
    def cost(self, t):
        return self.a*t+self.b

# Given action set, return num of agents on each highway, in the form of [h1, h2, h3, h4]
def num_on_highways(agent_acts):
    nums = [0, 0, 0, 0]
    for i in agent_acts:
        for j in i:
            nums[j] += 1
    return nums

# Given action set, return num of agents on each highway, in the form of [h1, h2, h3, h4], excluding agent i
def other_than_i_num_on_highways(i, agent_acts):
    nums = [0, 0, 0, 0]
    for k in range(len(agent_acts)):
        if k != i:
            for j in agent_acts[k]:
                nums[j] += 1
    return nums

# Calculate an agent's action given other agents' actions
def agent_cost(i, agent_acts, highways):
    agent_choice = agent_acts[i]
    current_num_on_highways = num_on_highways(agent_acts)
    cost = 0 
    for i in agent_choice:
        cost += highways[i].cost(current_num_on_highways[i])
    return cost

# Randomly assign agents with choices
def random_init(agent_num, choices):
    agent_acts = []
    for i in range(agent_num):
        r_index = random.randint(0, len(choices)-1)
        agent_acts.append(choices[r_index])
    return agent_acts

# Stats
def stats(agent_act, choices):
    nums = [0, 0, 0]
    for i in agent_act:
        if i == choices[0]:
            nums[0] += 1
        elif i == choices[1]:
            nums[1] += 1
        else:
            nums[2] += 1
    print(str(choices[0])+ ": " + str(nums[0]))
    print(str(choices[1])+ ": " + str(nums[1]))
    print(str(choices[2])+ ": " + str(nums[2]))

# Learning Model: Fictious Play
# Best Choice given other agents' play
def best_choice(i, agent_acts, choices, highways):
    current_num_on_highways = other_than_i_num_on_highways(i, agent_acts)
    costs = []
    for j in choices:
        cost = 0
        for k in j:
            current_num_on_highways[k] += 1
            cost += highways[k].cost(current_num_on_highways[k])
        costs.append([j, cost])
    # find min cost 
    min = costs[0][1]
    choice = 0
    for i in range(len(costs)):
        if costs[i][1] < min:
            choice = i
    return costs[choice]

# Repeatedly make best choices
def fictious_play(init_agent_acts, choices, highways):
    next_round = []
    changes = False
    for i in range(len(init_agent_acts)):
        its_best_choice = best_choice(i, init_agent_acts, choices, highways)
        # print("agent "+str(i)+"'s best choices is ")
        # print(its_best_choice)
        if len(its_best_choice[0]) != len(init_agent_acts[i]):
            changes = True
        else:
            for j in range(len(its_best_choice[0])):
                if its_best_choice[0][j] != init_agent_acts[i][j]:
                    changes = True
        next_round.append(its_best_choice[0])
        init_agent_acts[i] = its_best_choice[0]
    return [changes, next_round]

# Setup
#  4 Highways
h1 = highway(2,0)
h2 = highway(3,0)
h3 = highway(0,200)
h4 = highway(0,1000)
highways = [h1, h2, h3, h4]

# finals
choices = [[0, 2], [1, 2], [3]]

# fields
total_util = 0
agent_num = 1000
agent_acts = []
agent_util = []



# Main 
agent_acts = random_init(agent_num, choices)
print("")
print("Initial")
print(agent_acts)
change = True
round = 1
while change:
    re = fictious_play(agent_acts, choices, highways)
    change = re[0]
    agent_acts = re[1]
    print("")
    print("Round " + str(round))
    stats(agent_acts, choices)
    round += 1