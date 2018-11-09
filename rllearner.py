class RLLearner:

    def __init__(self, total_training_steps):
        self.total_training_steps = total_training_steps
        self.current_step = 0

    def keep_training(self):
        return self.current_step <= self.total_training_steps

    def start(self, environment, rl_agent):
        print("Starting learning ...")

        episode_finished = False
        environment.reset()

        while self.keep_training():

            if episode_finished:
                environment.reset()

            previous_state = environment.get_state()
            action_performed, new_state, episode_finished, reward = environment.step(rl_agent)

            rl_agent.observe_action_effects(previous_state=previous_state, action_performed=action_performed,
                                            reward=reward, new_state=new_state)

            self.current_step += 1
