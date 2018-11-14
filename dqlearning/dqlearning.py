import tensorflow as tf
import os
from tqdm import tqdm


class DeepQLearning(object):

    def __init__(self, logger, total_training_steps, decay_steps, train_frequency, batch_size, counter_for_learning,
                 transfer_frequency, save_frequency, checkpoint_path):
        self.total_training_steps = total_training_steps
        self.decay_steps = decay_steps

        self.train_frequency = train_frequency
        self.batch_size = batch_size

        self.counter_for_learning = counter_for_learning
        self.transfer_frequency = transfer_frequency
        self.save_frequency = save_frequency

        self.checkpoint_path = checkpoint_path
        self.logger = logger

        self.scope = 'train'
        self.training_step_var = self.get_global_step_variable()

    def get_global_step_variable(self):
        with tf.variable_scope(self.scope):
            global_step_variable = tf.Variable(0, trainable=False, name='training_step')

        return global_step_variable

    def train_agents(self, agent_wrappers, training_step, session):
        for agent_wrapper in agent_wrappers:
            self.logger.debug("Attempting training on agent " + agent_wrapper.name)

            agent_wrapper.train(session=session, batch_size=self.batch_size)

            if training_step % self.transfer_frequency:
                agent_wrapper.update_target_weights(session)

    def start(self, simulation_environment, agent_wrappers, enable_restore):

        with tqdm(total=self.total_training_steps) as progress_bar:

            with tf.Session() as session:
                initializer = tf.global_variables_initializer()
                saver = tf.train.Saver()

                checkpoint_file = self.checkpoint_path + ".index"
                if os.path.isfile(checkpoint_file) and enable_restore:
                    self.logger.info("Restoring checkpoint: " + checkpoint_file)
                    saver.restore(session, self.checkpoint_path)
                else:
                    session.run(initializer)

                episode_finished = False

                simulation_environment.global_counter = 0
                simulation_environment.session = session
                simulation_environment.reset(agent_wrappers)

                while True:
                    training_step = self.training_step_var.eval()
                    if training_step >= self.total_training_steps:
                        break

                    simulation_environment.global_counter += 1
                    if episode_finished:
                        self.logger.debug("Episode finished!")
                        for wrapper in agent_wrappers:
                            self.logger.debug(
                                "Agent: " + wrapper.name + " Episode Rewards: " + str(wrapper.episode_rewards))
                        simulation_environment.reset(agent_wrappers)

                    previous_state = simulation_environment.get_system_state()

                    self.logger.debug("Training step: %s  Global counter: %s",
                                      str(training_step), str(simulation_environment.global_counter))

                    actions_performed, new_state, episode_finished, rewards = simulation_environment.step(
                        rl_agents=agent_wrappers)

                    self.logger.debug("actions_performed %s new_state %s episode_finished %s rewards %s",
                                      actions_performed,
                                      new_state,
                                      episode_finished, rewards)

                    for agent_wrapper in agent_wrappers:
                        action_performed = actions_performed[agent_wrapper.name]
                        reward = rewards[agent_wrapper.name]
                        agent_wrapper.observe_action_effects(previous_state, action_performed, reward,
                                                             new_state)

                    if simulation_environment.global_counter > self.counter_for_learning \
                            and simulation_environment.global_counter % self.train_frequency == 0:
                        self.logger.debug(
                            "Triggering training: global_counter " + str(
                                simulation_environment.global_counter) + " counter_for_learning: " + str(
                                self.counter_for_learning) + " train_frequency " + str(self.train_frequency))
                        self.train_agents(agent_wrappers=agent_wrappers, training_step=training_step, session=session)
                        progress_bar.update(1)

                    if training_step % self.save_frequency:
                        self.logger.debug(
                            "Saving training progress at: " + self.checkpoint_path + " training_step " + str(
                                training_step))
                        saver.save(session, self.checkpoint_path)
