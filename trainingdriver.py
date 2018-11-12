import logging

import dqlagent
import dqlearning
import numpy as np

import environment

INPUT_NUMBER = 2
HIDDEN_UNITS = 24

MAX_STEPS = int(1e6)
NUM_OF_SERVERS = 10
PRIORITIES = np.arange(0, 4)
REWARDS = np.power(2, np.arange(0, 4))
FREE_PROBABILITY = 0.06


def main():
    logging_mode = 'w'
    enable_restore = False

    logging_level = logging.DEBUG

    total_training_steps = 10000
    decay_steps = int(total_training_steps / 2)

    train_frequency = 4
    batch_size = 32

    discount_factor = 0.99
    learning_rate = 1e-4

    counter_for_learning = int(total_training_steps / 100)
    transfer_frequency = counter_for_learning
    save_frequency = counter_for_learning * 0.1

    initial_epsilon = 1.0
    final_epsilon = 0.1

    replay_memory_size = 100

    scenario = "access_control_"
    log_filename = scenario + '_tech_debt_rl.log'
    checkpoint_path = './chk/' + scenario + '.ckpt'

    logger = logging.getLogger(scenario + "-DQNetwork-Training->")
    handler = logging.FileHandler(log_filename, mode=logging_mode)
    logger.addHandler(handler)
    logger.setLevel(logging_level)

    dq_learner = dqlearning.DeepQLearning(logger=logger, total_training_steps=total_training_steps,
                                          decay_steps=decay_steps,
                                          train_frequency=train_frequency, batch_size=batch_size,
                                          counter_for_learning=counter_for_learning,
                                          transfer_frequency=transfer_frequency,
                                          save_frequency=save_frequency,
                                          checkpoint_path=checkpoint_path)

    dev_name = "dq_learner"
    developer_agent = dqlagent.DeepQLearner(name=dev_name,
                                            learning_rate=learning_rate,
                                            actions=environment.ACTIONS,
                                            discount_factor=discount_factor,
                                            counter_for_learning=counter_for_learning,
                                            input_number=INPUT_NUMBER, hidden_units=HIDDEN_UNITS,
                                            logger=logger,
                                            initial_epsilon=initial_epsilon, final_epsilon=final_epsilon,
                                            decay_steps=decay_steps,
                                            replay_memory_size=replay_memory_size,
                                            global_step=dq_learner.training_step_var)

    access_control_environment = environment.AccessControlEnvironment(logger=logger,
                                                                      max_steps=MAX_STEPS,
                                                                      num_of_servers=NUM_OF_SERVERS,
                                                                      priorities=PRIORITIES,
                                                                      rewards=REWARDS,
                                                                      free_probability=FREE_PROBABILITY)

    dq_learner.start(access_control_environment, [developer_agent], enable_restore)


if __name__ == "__main__":
    main()
