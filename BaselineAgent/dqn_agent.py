from utility.environment_interface import EnvironmentInterface
from collections import deque
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
observation_dict_to_tensor_mapping = {
    str({'type': 'sensing', 'is_sensed': {0: False, 1: False}}): np.array([0, 0]),
    str({'type': 'sensing', 'is_sensed': {0: True, 1: False}}): np.array([1, 0]),
    str({'type': 'sensing', 'is_sensed': {0: False, 1: True}}): np.array([0, 1]),
    str({'type': 'sensing', 'is_sensed': {0: True, 1: True}}): np.array([1, 1]),
    str({'type': 'tx_data_packet', 'success_freq_channel_list': []}): np.array([2, 2]),
    str({'type': 'tx_data_packet', 'success_freq_channel_list': [0]}): np.array([3, 2]),
    str({'type': 'tx_data_packet', 'success_freq_channel_list': [1]}): np.array([2, 3]),
    str({'type': 'tx_data_packet', 'success_freq_channel_list': [0, 1]}): np.array([3, 3])
}
action_index_to_dict_mapping = {
    0: {'type': 'sensing'},
    1: {'type': 'tx_data_packet', 'freq_channel_list': [0], 'num_unit_packet': 1},
    2: {'type': 'tx_data_packet', 'freq_channel_list': [1], 'num_unit_packet': 1},
    3: {'type': 'tx_data_packet', 'freq_channel_list': [0], 'num_unit_packet': 2},
    4: {'type': 'tx_data_packet', 'freq_channel_list': [1], 'num_unit_packet': 2},
    5: {'type': 'tx_data_packet', 'freq_channel_list': [0, 1], 'num_unit_packet': 1},
    6: {'type': 'tx_data_packet', 'freq_channel_list': [0, 1], 'num_unit_packet': 2},
}

class Agent:
    def __init__(self, environment, unit_packet_success_reward, unit_packet_failure_reward, discount_factor, dnn_learning_rate,
                 initial_epsilon, epsilon_decay, min_epsilon,):
        self._env = environment
        self._unit_packet_success_reward = unit_packet_success_reward
        self._unit_packet_failure_reward = unit_packet_failure_reward
        self._discount_factor = discount_factor
        self._dnn_learning_rate = dnn_learning_rate
        self._num_freq_channel = 2
        self._num_action = 7
        self._epsilon = initial_epsilon
        self._epsilon_decay = epsilon_decay
        self._min_epsilon = min_epsilon
        self._replay_memory = deque()
        self._observation = np.zeros(self._num_freq_channel)
        self._model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(self._num_action, activation='relu', kernel_initializer='glorot_normal'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self._dnn_learning_rate))
        return model

    def set_init(self, run_time):
        self._env.start_simulation(time_us=run_time)
        initial_action = {'type': 'sensing'}
        observation_dict = self._env.step(initial_action)
        self._observation = observation_dict_to_tensor_mapping[str(observation_dict)]


    def train(self, run_time, dnn_epochs):
        self._env.disable_video_logging()
        self._env.disable_text_logging()
        self.set_init(run_time)
        self._replay_memory.clear()
        print("Replay memory stack")
        while True:
            sim_finish = self.accumulate_replay_memory(self._epsilon)
            if sim_finish:
                break
        observation, reward = self.replay()
        self._model.fit(observation, reward, epochs=dnn_epochs)
        self._model.save_weights('my_model')
        print(f"(epsilon: {self._epsilon})")
        self._epsilon = max(self._epsilon * self._epsilon_decay, self._min_epsilon)

    def test(self, run_time: int):
        self._env.enable_video_logging()
        self._env.enable_text_logging()
        self.set_init(run_time)
        self._model.load_weights('my_model')
        while True:
            action, _, _ = self.get_dnn_action_and_value(self._observation)
            action_dict = action_index_to_dict_mapping[int(action)]
            observation_dict = self._env.step(action_dict)
            if observation_dict == 0:
                break
            self._observation = observation_dict_to_tensor_mapping[str(observation_dict)]
            print(f"{self._env.get_score()}\r", end='', flush=True)

    def get_dnn_action_and_value(self, observation):
        if observation.ndim == 1:
            observation = observation[np.newaxis, ...]
        action_value = self._model.predict(observation)
        best_action = np.argmax(action_value, axis=1)
        best_value = np.amax(action_value, axis=1)
        return best_action, best_value, action_value

    def accumulate_replay_memory(self, random_prob):
        if np.random.rand() < random_prob:  # epsilon
            observation_dict = self._env.random_action_step()
            if observation_dict == {}:
                return True
            observation = observation_dict_to_tensor_mapping[str(observation_dict)]
            action, _, _ = self.get_dnn_action_and_value(observation)
        else:
            action, _, _ = self.get_dnn_action_and_value(self._observation)
        action_dict = action_index_to_dict_mapping[int(action)]
        observation_dict = self._env.step(action_dict)
        if observation_dict == 0:
            return True
        else:
            reward = self.get_reward(action_dict, observation_dict)
            next_observation = observation_dict_to_tensor_mapping[str(observation_dict)]
            experience = (self._observation, action, reward, next_observation)
            self._replay_memory.append(experience)
            self._observation = next_observation

    def get_reward(self, action, observation):
        observation_type = observation['type']
        reward = 0
        if observation_type == 'sensing':
            reward = 0
        elif observation_type == 'tx_data_packet':
            num_tx_packet = len(action['freq_channel_list'])
            num_success_packet = len(observation['success_freq_channel_list'])
            num_failure_packet = num_tx_packet - num_success_packet
            reward = num_success_packet * self._unit_packet_success_reward + num_failure_packet * self._unit_packet_failure_reward
        return reward

    def replay(self):
        observation = np.stack([x[0] for x in  self._replay_memory], axis=0)
        next_observation = np.stack([x[3] for x in  self._replay_memory], axis=0)
        _, _, action_reward = self.get_dnn_action_and_value(observation)
        _, future_reward, _ = self.get_dnn_action_and_value(next_observation)
        for ind, sample in enumerate(self._replay_memory):
            action = sample[1]
            immediate_reward = sample[2]
            action_reward[ind, action] = immediate_reward + self._discount_factor * future_reward[ind]
        return observation, action_reward

if __name__ == "__main__":
    env = EnvironmentInterface()
    env.connect()
    agent = Agent(environment=env, unit_packet_success_reward=1, unit_packet_failure_reward=-2, discount_factor=0.9,
                  dnn_learning_rate=0.001, initial_epsilon=1, epsilon_decay=0.95, min_epsilon=0.1)
    agent.train(run_time=50000, dnn_epochs=1)
    agent.test(10000)