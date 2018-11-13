import rllearner
import rlagent
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ACCEPT_ACTION = 1
REJECT_ACTION = 0

ACTIONS = [REJECT_ACTION, ACCEPT_ACTION]


class AccessControlEnvironment:
    """
    Simulation environment for the Access-Control Queuing Task, included in the Sutton book as Example 10.2.
    """

    def __init__(self, max_steps, num_of_servers, priorities, rewards, free_probability, logger):
        self.logger = logger
        self.max_steps = max_steps
        self.num_of_servers = num_of_servers
        self.priorities = priorities
        self.rewards = rewards
        self.free_probability = free_probability

        self.current_free_servers = None
        self.current_step = None
        self.current_priority = None

    def reset(self, agents):
        self.current_step = 0
        self.set_system_state(current_free_servers=self.num_of_servers,
                              current_priority=np.random.choice(self.priorities))

        for agent in agents:
            agent.reset()

    def enact_action(self, selected_action):
        if self.current_free_servers > 0 and selected_action == ACCEPT_ACTION:
            self.current_free_servers -= 1

        reward = self.rewards[self.current_priority] * selected_action

        busy_servers = self.num_of_servers - self.current_free_servers
        available_servers = self.current_free_servers + np.random.binomial(busy_servers, self.free_probability)

        self.set_system_state(current_free_servers=available_servers,
                              current_priority=np.random.choice(self.priorities))

        return reward

    def set_system_state(self, current_free_servers, current_priority):
        self.current_free_servers = current_free_servers
        self.current_priority = current_priority

    def get_system_state(self):
        return self.current_free_servers, self.current_priority

    def step(self, rl_agents, **_):
        self.logger.debug(
            "Current step: " + str(self.current_step) + " Free servers: " + str(self.current_free_servers))

        actions = {}
        rewards = {}

        for rl_agent in rl_agents:
            selected_action = rl_agent.select_action(environment=self)
            reward = self.enact_action(selected_action)

            actions[rl_agent.name] = selected_action
            rewards[rl_agent.name] = reward

        self.current_step += 1

        episode_finished = self.current_step >= self.max_steps
        new_state = self.get_system_state()

        return actions, new_state, episode_finished, rewards


def plot_policy(rl_learner, environment, filename):
    policy_data = np.zeros((len(environment.priorities), environment.num_of_servers + 1))

    for priority in environment.priorities:
        for free_servers in range(environment.num_of_servers + 1):
            environment.set_system_state(current_free_servers=free_servers, current_priority=priority)
            selected_action = rl_learner.select_action(environment=environment)
            policy_data[priority, free_servers] = selected_action

    ax = sns.heatmap(policy_data, cmap="YlGnBu", xticklabels=range(environment.num_of_servers + 1),
                     yticklabels=environment.priorities)
    ax.set_title('Policy (0 Reject, 1 Accept)')
    ax.set_xlabel('Number of free servers')
    ax.set_ylabel('Priority')

    plt.savefig(filename)
    plt.close()

    print("Plot saved at: " + filename)


def main():
    max_steps = int(1e6)

    num_of_servers = 10
    priorities = np.arange(0, 4)
    rewards = np.power(2, np.arange(0, 4))
    free_probability = 0.06

    queueing_environment = AccessControlEnvironment(max_steps=max_steps, num_of_servers=num_of_servers,
                                                    priorities=priorities,
                                                    rewards=rewards, free_probability=free_probability)
    rl_learner = rllearner.RLLearner(total_training_steps=max_steps)
    rl_agent = rlagent.RLAgent(actions=[REJECT_ACTION, ACCEPT_ACTION])
    # rl_learner.start(environment=queueing_environment, rl_agent=rl_agent)

    filename = 'policy.png'
    plot_policy(rl_agent, queueing_environment, filename=filename)


if __name__ == "__main__":
    main()
