import math
from scipy import special
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt
import env_access
import os


class Node:
    def __init__(self, index):
        self.index = index
        self.state = []  # The list of local state at different time steps
        self.action = []  # The list of local actions at different time steps
        self.reward = []  # The list of local actions at different time steps
        self.currentTimeStep = 0  # Record the current time step.
        self.paramsDict = {}  # use a hash map to query the parameters given a state (or neighbors' states)
        self.QDict = {}  # use a hash map to query to the Q value given a (state, action) pair
        self.kHop = []  # The list to record the (state, action) pairs of k-hop neighbors

    # get the local state at timeStep
    def get_state(self, time_step):
        if time_step <= len(self.state) - 1:
            return self.state[time_step]
        else:
            print("getState: In node ", self.index, ", timeStep overflows!")
            return -1

    # get the local action at timeStep
    def get_action(self, time_step):
        if time_step <= len(self.action) - 1:
            return self.action[time_step]
        else:
            print("getAction: In node ", self.index, ", timeStep overflows!")
            return -1

    # get the local reward at timeStep
    def get_reward(self, time_step):
        if time_step <= len(self.reward) - 1:
            return self.reward[time_step]
        else:
            print("getReward: In node ", self.index, ", timeStep overflows!")
            return -1

    # get the kHopStateAction at timeStep
    def get_k_hop_state_action(self, time_step):
        if time_step <= len(self.kHop) - 1:
            return self.kHop[time_step]
        else:
            print("getKHopStateAction: In node ", self.index, ", timeStep overflows!")
            return -1

    # get the local Q at timeStep
    def get_q(self, k_hop_state_action):
        # if the Q value of kHopStateAction hasn't been queried before, return 0.0 (initial value)
        return self.QDict.get(k_hop_state_action, 0.0)

    # initialize the local state
    def initialize_state(self):
        pass

    # update the local state, it may depends on the states of other nodes at the last time step.
    # Remember to increase self.currentTimeStep by 1
    def update_state(self):
        pass

    # update the local action
    def update_action(self):
        pass

    # update the local reward
    def update_reward(self):
        pass

    # update the local Q value
    def update_q(self):
        pass

    # update the local parameter
    def update_params(self):
        pass

    # clear the record. Called when a new inner loop starts.
    def restart(self):
        self.state.clear()
        self.action.clear()
        self.reward.clear()
        self.kHop.clear()
        self.currentTimeStep = 0


class AccessNode(Node):
    def __init__(self, index, deadline, arrival_prob, gamma, k, env):
        super(AccessNode, self).__init__(index)
        self.ddl = deadline  # the initial deadline of each packet
        self.k = k
        self.arrivalProb = arrival_prob  # the arrival probability at each timestep
        self.gamma = gamma  # the discount factor
        # we use packetQueue to represent the current local state, which is (e_1, e_2, ..., e_d)
        self.packetQueue = np.zeros(self.ddl,
                                    dtype=int)  # use 1 to represent a packet with this remaining time, otherwise 0
        self.accessPoints = env.accessNetwork.find_access(
            i=index)  # find and cache the access points this node can access
        self.accessNum = len(self.accessPoints)  # the number of access points
        self.actionNum = self.accessNum * self.ddl + 1  # the number of possible actions, action is a tuple (slot, accessPoint)
        # construct a list of possible actions
        self.actionList = [(-1, -1)]  # (-1, -1) is an empty action that does nothing
        for slot in range(self.ddl):
            for a in self.accessPoints:
                self.actionList.append((slot, a))

        self.env = env

    # initialize the local state (called at the beginning of the training process)
    def initialize_state(self):
        self.state.append(self.env.observe_state_g(self.index, 0)[0])  # append this state to state record

    # At each time step t, call updateState, updateAction, updateReward, updateQ in this order
    def update_state(self):
        self.currentTimeStep += 1
        self.state.append(self.env.observe_state_g(self.index, 0)[0])  # append this state to state record
        self.packetQueue = self.state[-1]

    def update_action(self, benchmark_policy=None):
        if benchmark_policy is not None:
            actProb = benchmark_policy[0]
            flagAct = np.random.binomial(1, actProb)  # should I send out a packet?
            if flagAct == 0:
                self.action.append((-1, -1))
                self.env.update_action(self.index, (-1, -1))
                return
            # find the packet with the earliest ddl
            benchSlot = -1
            for i in range(self.ddl):
                if self.packetQueue[i] > 0:
                    benchSlot = i
                    break
            if benchSlot == -1:
                self.action.append((-1, -1))
                env.update_action(self.index, (-1, -1))
                return
            # select the access point to send to
            benchProb = benchmark_policy[1:]
            benchAccessPoint = self.accessPoints[np.random.choice(a=self.accessNum, p=benchProb)]
            self.action.append((benchSlot, benchAccessPoint))
            self.env.update_action(self.index, (benchSlot, benchAccessPoint))
            return
        # get the current state
        currentState = self.state[-1]

        # fetch the params based on the current state. If haven't updated before, return all zeros
        params = self.paramsDict.get(currentState, np.zeros(self.actionNum))
        # compute the probability vector
        probVec = special.softmax(params)
        # randomly select an action based on probVec
        currentAction = self.actionList[np.random.choice(a=self.actionNum, p=probVec)]
        self.action.append(currentAction)
        self.env.update_action(self.index, currentAction)

    # oneHopNeighbors is a list of accessNodes
    def update_reward(self):
        currentReward = self.env.observe_reward(self.index)
        self.reward.append(currentReward)

    # need to call this after the first time step
    def update_k_hop(self):
        self.kHop.append(self.env.observe_state_action_g(self.index, self.k))

    # kHopNeighbors is a list of accessNodes, alpha is learning rate
    def update_q(self, alpha):
        lastStateAction = self.kHop[-1]
        currentStateAction = self.env.observe_state_action_g(self.index, self.k)
        # fetch the Q value based on neighbors' states and actions
        lastQTerm1 = self.QDict.get(lastStateAction, 0.0)
        lastQTerm2 = self.QDict.get(currentStateAction, 0.0)
        # compute the temporal difference
        temporalDiff = self.reward[-2] + self.gamma * lastQTerm2 - lastQTerm1
        # perform the Q value update
        self.QDict[lastStateAction] = lastQTerm1 + alpha * temporalDiff
        # if this time step 1, we should also put lastStateAction into history record
        if len(self.kHop) == 0:
            self.kHop.append(lastStateAction)
        # put currentStateAction into history record
        self.kHop.append(currentStateAction)

    # eta is the learning rate
    def update_params(self, k_hop_neighbors, eta):
        # for t = 0, 1, ..., T, compute the term in g_{i, t}(m) before \nabla
        mutiplier1 = np.zeros(self.currentTimeStep + 1)
        for neighbor in k_hop_neighbors:
            for t in range(self.currentTimeStep + 1):
                neighborKHop = neighbor.get_k_hop_state_action(t)
                neighborQ = neighbor.get_q(neighborKHop)
                mutiplier1[t] += neighborQ
        for t in range(self.currentTimeStep + 1):
            mutiplier1[t] *= pow(self.gamma, t)
            mutiplier1[t] /= nodeNum
        # finish constructing mutiplier1

        # compute the gradient with respect to the parameters associated with s_i(t)
        for t in range(self.currentTimeStep + 1):
            currentState = self.state[t]
            currentAction = self.action[t]
            params = self.paramsDict.get(currentState, np.zeros(self.actionNum))
            probVec = special.softmax(params)
            grad = -probVec
            actionIndex = self.actionList.index(currentAction)  # get the index of currentAction
            grad[actionIndex] += 1.0
            self.paramsDict[currentState] = params + eta * mutiplier1[t] * grad

    # compute the total discounted reward
    def total_reward(self):
        totalReward = 0.0
        for t in range(self.currentTimeStep):
            totalReward += (pow(self.gamma, t) * self.reward[t])
        return totalReward


# do not update Q when evaluating a policy
def eval_policy(node_list, rounds, env):
    totalRewardSum = 0.0
    for _ in range(rounds):
        env.initialize()
        for i in range(nodeNum):
            node_list[i].restart()
            node_list[i].initialize_state()

        for i in range(nodeNum):
            node_list[i].update_action()
        env.generate_reward()
        for i in range(nodeNum):
            node_list[i].update_reward()

        for t in range(1, T + 1):
            env.step()
            for i in range(nodeNum):
                node_list[i].update_state()
            for i in range(nodeNum):
                node_list[i].update_action()
            env.generate_reward()
            for i in range(nodeNum):
                node_list[i].update_reward()
        # compute the total reward
        averageReward = 0.0
        for i in range(nodeNum):
            averageReward += node_list[i].total_reward()
        averageReward /= nodeNum
        totalRewardSum += averageReward
    return totalRewardSum / rounds


def evalBenchmark(node_list, rounds, act_prob, env):
    totalRewardSum = 0.0
    benchmarkPolicyList = []
    for i in range(nodeNum):
        accessPoints = env.accessNetwork.find_access(i)
        accessPointsNum = len(accessPoints)
        benchmarkPolicy = np.zeros(accessPointsNum + 1)
        totalSum = 0.0
        for j in range(accessPointsNum):
            # print(accessNetwork.serviceNum[accessPoints[j]])
            tmp = 100 * env.accessNetwork.transmitProb[accessPoints[j]] / env.accessNetwork.serviceNum[accessPoints[j]]
            totalSum += tmp
            benchmarkPolicy[j + 1] = tmp
        for j in range(accessPointsNum):
            benchmarkPolicy[j + 1] /= totalSum
        benchmarkPolicy[0] = act_prob
        benchmarkPolicyList.append(benchmarkPolicy)

    for _ in range(rounds):
        env.initialize()
        for i in range(nodeNum):
            node_list[i].restart()
            node_list[i].initialize_state()

        for i in range(nodeNum):
            node_list[i].update_action(benchmarkPolicyList[i])
        env.generate_reward()
        for i in range(nodeNum):
            node_list[i].update_reward()

        for t in range(1, T + 1):
            env.step()
            for i in range(nodeNum):
                node_list[i].update_state()
            for i in range(nodeNum):
                node_list[i].update_action(benchmarkPolicyList[i])
            env.generate_reward()
            for i in range(nodeNum):
                node_list[i].update_reward()
        # compute the total reward
        averageReward = 0.0
        for i in range(nodeNum):
            averageReward += node_list[i].total_reward()
        averageReward /= nodeNum
        totalRewardSum += averageReward

    return totalRewardSum / rounds


if __name__ == "__main__":
    k = 1
    height = 3
    width = 4
    node_per_grid = 2
    nodeNum = height * width * node_per_grid
    env = env_access.AccessGridEnv(height=height, width=width, k=k, node_per_grid=node_per_grid)

    gamma = 0.7
    ddl = 2
    arrivalProb = 0.5
    T = 10
    M = 120000

    evalInterval = 2000     # evaluate the policy every evalInterval rounds (outer loop)
    restartInterval = 100
    accessNodeList = []
    for i in range(nodeNum):
        accessNodeList.append(AccessNode(index=i, deadline=ddl, arrival_prob=arrivalProb, gamma=gamma, k=k, env=env))

    script_dir = os.path.dirname(__file__)

    with open(script_dir+'data/Tabular-Access-{}-{}-{}.txt'.format(height, width, k), 'w') as f:  # used to check the progress of learning
        # first erase the file
        f.seek(0)
        f.truncate()

    policyRewardList = []
    for m in trange(M):
        if m == 0:
            policyRewardList.append(eval_policy(node_list=accessNodeList, rounds=400, env=env))
            with open(script_dir+'data/Tabular-Access-{}-{}-{}.txt'.format(height, width, k), 'w') as f:
                f.write("%f\n" % policyRewardList[-1])
            # first find the best benchmark policy and its discounted reward
            bestBenchmark = 0.0
            bestBenchmarkProb = 0.0
            for i in range(20):
                tmp = evalBenchmark(node_list=accessNodeList, rounds=100, act_prob=i / 20.0, env=env)
                if tmp > bestBenchmark:
                    bestBenchmark = tmp
                    bestBenchmarkProb = i / 20.0
            print(bestBenchmark, bestBenchmarkProb)

        env.initialize()
        for i in range(nodeNum):
            accessNodeList[i].restart()
            accessNodeList[i].initialize_state()
        for i in range(nodeNum):
            accessNodeList[i].update_action()
        env.generate_reward()
        for i in range(nodeNum):
            accessNodeList[i].update_reward()
        for i in range(nodeNum):
            accessNodeList[i].update_k_hop()
        # start inner loop
        for t in range(1, T + 1):
            env.step()
            for i in range(nodeNum):
                accessNodeList[i].update_state()
            for i in range(nodeNum):
                accessNodeList[i].update_action()
            env.generate_reward()
            for i in range(nodeNum):
                accessNodeList[i].update_reward()
            for i in range(nodeNum):
                accessNodeList[i].update_q(1.0 / math.sqrt((m % restartInterval) * T + t))
        # end inner loop

        # perform the grad update
        for i in range(nodeNum):
            neighborList = []
            for j in env.accessNetwork.find_neighbors(i, k):
                neighborList.append(accessNodeList[j])
            accessNodeList[i].update_params(neighborList, 5.0 / math.sqrt(m % restartInterval + 1))

        # perform a policy evaluation
        if m % evalInterval == evalInterval - 1:
            tempReward = eval_policy(node_list=accessNodeList, rounds=400, env=env)
            with open(script_dir+'data/Tabular-Access-{}-{}-{}.txt'.format(height, width, k), 'a') as f:
                f.write("%f\n" % tempReward)
            policyRewardList.append(tempReward)

    lam = np.linspace(0, (len(policyRewardList) - 1) * evalInterval, len(policyRewardList))
    plt.plot(lam, policyRewardList)
    plt.hlines(y=bestBenchmark, xmin=0, xmax=M, colors='g', label="Benchmark")
    plt.savefig(script_dir+"data/Tabular-Access-{}-{}-{}.jpg".format(height, width, k))