import socket
import logging
import random
import time
from .messenger import Messenger
from typing import Optional, Dict, List, Union

logging.basicConfig(level=logging.INFO, format='%(message)s')


class EnvironmentInterface:
    def __init__(self):
        self._identifier = 'p1'
        self._server_address: str = '127.0.0.1'
        self._server_port: int = 8888
        self._socket: socket = None
        self._messenger: Optional[Messenger] = None
        self._sta_list: Optional[List] = None
        self._freq_channel_list: Optional[List] = None
        self._num_unit_packet_list: Optional[List] = None
        self._first_step: bool = True
        self._state: str = 'unconnected'
        self._score: Dict = {'packet_success': 0, 'collision': 0, 'total_score': 0}

    @property
    def state(self):
        return self._state

    @property
    def freq_channel_list(self):
        return self._freq_channel_list

    @property
    def num_unit_packet_list(self):
        return self._num_unit_packet_list

    def connect(self):
        if self._state == 'unconnected':
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            socket_connected = False
            for trial in range(10):
                try:
                    logging.warning(f'Try to connect to the environment docker ({trial+1}/10).')
                    self._socket.connect((self._server_address, self._server_port))
                except socket.error:
                    logging.warning('Connection failed.')
                    time.sleep(0.5)
                else:
                    socket_connected = True
                    break
            if socket_connected:
                self._messenger = Messenger(self._socket)
                self._messenger.send('player_id', self._identifier)
                msg_type, msg = self._messenger.recv()
                if msg_type == 'connection_successful':
                    self._state = 'idle'
                    logging.info('Successfully connected to environment.')
                else:
                    self._environment_failed()
            else:
                logging.warning('Check out if the environment docker is up.')
        else:
            logging.info('Already connected to environment.')

    def start_simulation(self, time_us):
        if not ((isinstance(time_us, float) or isinstance(time_us, int)) and time_us > 0):
            logging.warning('Simulation time should be a positive number.')
            return
        if self._state == 'idle':
            self._messenger.send('start_simulation', time_us)
            msg_type, msg = self._messenger.recv()
            if msg_type == 'operator_info':
                self._sta_list = msg['sta_list']
                self._freq_channel_list = msg['freq_channel_list']
                self._num_unit_packet_list = msg['num_unit_packet_list']
                self._first_step = True
                self._score = {'packet_success': 0, 'collision': 0, 'total_score': 0}
                self._state = 'running'
                logging.info('Simulation started.')
            else:
                self._environment_failed()
        elif self._state == 'unconnected':
            logging.info("Environment is not connected.")
        elif self._state == 'running':
            logging.info("Simulation is already running.")

    def step(self, action: Dict) -> Union[Optional[Dict], int]:
        if self._first_step:
            self._messenger.recv()  # Discard the first observation
            self._first_step = False
        if self._check_action(action):
            self._messenger.send('action', action)
        else:
            return -1
        msg_type, msg = self._messenger.recv()
        if msg_type == 'observation':
            self._score = msg['score']
            observation = msg['observation']
            if 'is_sensed' in observation:
                is_sensed = {}
                for ch in observation['is_sensed']:
                    is_sensed[int(ch)] = (observation['is_sensed'][ch] == 1)
                observation['is_sensed'] = is_sensed
            return observation
        elif msg_type == 'simulation_finished':
            logging.info("Simulation is finished.")
            self._state = 'idle'
            return 0
        else:
            self._environment_failed()

    def get_score(self) -> Dict:
        return self._score

    def _check_action(self, action: Dict) -> bool:
        if not isinstance(action, Dict):
            logging.warning("Action should be a dictionary.")
            return False
        if 'type' not in action:
            logging.warning("Key \'type\' should exist.")
            return False
        if action['type'] == 'sensing':
            return True
        elif action['type'] == 'tx_data_packet':
            if ('freq_channel_list' not in action) or ('num_unit_packet' not in action):
                logging.warning("Keys \'freq_channel_list\' and \'num_unit_packet\' should exist "
                                "if the type is \'tx_data_packet\'.")
                return False
            freq_channel_list = action['freq_channel_list']
            set_freq_channel_list = set(freq_channel_list)
            if len(set_freq_channel_list) == 0:
                logging.warning("There should be at least one frequency channel in \'freq_channel_list\'.")
                return False
            if len(set_freq_channel_list) < len(freq_channel_list):
                logging.warning("Value of \'freq_channel_list\' should be a list of unique channels.")
                return False
            if not set_freq_channel_list.issubset(set(self._freq_channel_list)):
                logging.warning(f"Value of \'freq_channel_list\' should be in {self._freq_channel_list}.")
                return False
            num_unit_packet = action['num_unit_packet']
            if not isinstance(num_unit_packet, int):
                logging.warning("Value of \'num_unit_packet\' should be an integer.")
                return False
            if action['num_unit_packet'] not in self._num_unit_packet_list:
                logging.warning(f"Value of \'num_unit_packet\' should be one of {self._num_unit_packet_list}.")
                return False
        else:
            logging.warning("Value of \'type\' should be either \'sensing\' or \'tx_data_packet\'.")
            return False
        return True

    def random_action_step(self, num_step=1):
        if self._state != 'running':
            logging.warning("Simulation is not running.")
        else:
            last_observation = {}
            for step in range(num_step):
                action_type = random.choice(['sensing', 'tx_data_packet'])
                if action_type == 'tx_data_packet':
                    num_channel = random.randint(1, len(self._freq_channel_list))
                    freq_channel_list = sorted(random.sample(self._freq_channel_list, num_channel))
                    num_unit_packet = random.choice(self._num_unit_packet_list)
                    action = {'type': 'tx_data_packet', 'freq_channel_list': freq_channel_list,
                              'num_unit_packet': num_unit_packet}
                else:
                    action = {'type': 'sensing'}
                logging.info(f"Action: {action}")
                observation = self.step(action)
                if observation == 0:
                    break
                else:
                    logging.info(f"Observation: {observation}")
                    logging.info(f"Score: {self._score}")
                    last_observation=observation
            return last_observation

    def _configure_logging(self, log_type: str, enable: bool):
        if self._state == 'idle':
            self._messenger.send('configure_logging', {'log_type': log_type, 'enable': enable})
            msg_type, msg = self._messenger.recv()
            if msg_type == 'logging_configured':
                logging.info("Logging is successfully configured.")
            else:
                self._environment_failed()
        elif self._state == 'running':
            logging.warning("Logging cannot be configured when the simulator is running.")
        elif self._state == 'unconnected':
            logging.warning("Environment is not connected.")

    def enable_video_logging(self):
        self._configure_logging(log_type='video', enable=True)

    def disable_video_logging(self):
        self._configure_logging(log_type='video', enable=False)

    def enable_text_logging(self):
        self._configure_logging(log_type='text', enable=True)

    def disable_text_logging(self):
        self._configure_logging(log_type='text', enable=False)

    def _environment_failed(self):
        raise Exception('Environment failed. Restart the environment docker.')

