import rllearner
import rlagent
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ACCEPT_ACTION = 1
REJECT_ACTION = 0


class QueuingEnvironment:

    def __init__(self, max_steps, num_of_servers, priorities, rewards, free_probability):
        self.max_steps = max_steps
        self.num_of_servers = num_of_servers
        self.priorities = priorities
        self.rewards = rewards
        self.free_probability = free_probability

        self.current_free_servers = None
        self.current_step = None
        self.current_priority = None

    def reset(self):
        self.current_free_servers = self.num_of_servers
        self.current_step = 0
        self.current_priority = np.random.choice(self.priorities)

    def enact_action(self, selected_action):
        if self.current_free_servers > 0 and selected_action == ACCEPT_ACTION:
            self.current_free_servers -= 1

        reward = self.rewards[self.current_priority] * selected_action

        busy_servers = self.num_of_servers - self.current_free_servers
        self.current_free_servers += np.random.binomial(busy_servers, self.free_probability)
        self.current_priority = np.random.choice(self.priorities)

        return reward

    def get_state(self):
        return self.current_free_servers, self.current_priority

    def step(self, rl_agent):
        print("Current step: " + str(self.current_step) + " Free servers: " + str(self.current_free_servers))
        system_state = self.get_state()
        selected_action = rl_agent.select_action(system_state=system_state)

        reward = self.enact_action(selected_action)

        self.current_step += 1

        episode_finished = self.current_step >= self.max_steps
        new_state = self.get_state()

        return selected_action, new_state, episode_finished, reward


def plot_policy(rl_learner, environment, filename):
    policy_data = np.zeros((len(environment.priorities), environment.num_of_servers + 1))

    for priority in environment.priorities:
        for free_servers in range(environment.num_of_servers + 1):
            selected_action = rl_learner.select_action(system_state=(free_servers, priority))
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
    # max_steps = int(1e6)
    max_steps = 1000

    num_of_servers = 10
    priorities = np.arange(0, 4)
    rewards = np.power(2, np.arange(0, 4))
    free_probability = 0.06

    queueing_environment = QueuingEnvironment(max_steps=max_steps, num_of_servers=num_of_servers, priorities=priorities,
                                              rewards=rewards, free_probability=free_probability)
    rl_learner = rllearner.RLLearner(total_training_steps=max_steps)
    rl_agent = rlagent.RLAgent(actions=[REJECT_ACTION, ACCEPT_ACTION])
    # rl_learner.start(environment=queueing_environment, rl_agent=rl_agent)

    filename = 'policy.png'
    plot_policy(rl_agent, queueing_environment, filename=filename)


if __name__ == "__main__":
    main()
