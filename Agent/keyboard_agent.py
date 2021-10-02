from utility.environment_interface import EnvironmentInterface
import numpy as np

class keyboardAgent:
    def __init__(self, environment: EnvironmentInterface, num_unit_packet):
        self._env = environment
        self._num_unit_packet = num_unit_packet
        self._freq_channel_list = []
        self._num_freq_channel = 0

    def run(self, run_time):
        self._env.start_simulation(time_us=run_time)
        self._freq_channel_list = self._env.freq_channel_list
        self._num_freq_channel = len(self._freq_channel_list)
        print('<Action Type>')
        print('sensing : s or S')
        print('tx_data_packet : (0 ~ {})\n'.format(self._num_freq_channel-1))

        while True:
            key = input('Action : ')
            action = self.keyboard_control(key=key)
            print("Action =", action)
            observation = self._env.step(action=action)
            print("Observation =", observation, "\n")

    def keyboard_control(self, key):
        channel_list = []
        if len(key) == 1:
            if key == 's' or key == 'S':
                action = {'type': 'sensing'}
            else:
                for i in range(self._num_freq_channel):
                    if int(key) == i:
                        channel_list.append(i)
                action = {'type': 'tx_data_packet', 'freq_channel_list': channel_list,
                          'num_unit_packet': self._num_unit_packet}
        else:
            key_list = key.split(',')
            int_key_list = list(map(int, key_list))
            int_key_list = np.array(int_key_list)
            channel_list = list(int_key_list)
            channel_list = [int(freq_channel) for freq_channel in channel_list]
            action = {'type': 'tx_data_packet', 'freq_channel_list': channel_list,
                      'num_unit_packet': self._num_unit_packet}
        return action


if __name__ == "__main__":
    env = EnvironmentInterface()
    env.connect()
    agent = keyboardAgent(environment=env, num_unit_packet=1)
    agent.run(100000)