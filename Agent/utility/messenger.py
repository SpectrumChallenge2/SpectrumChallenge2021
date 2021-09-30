import json
import struct


class Messenger:
    def __init__(self, sock):
        self._sock = sock
        self._tx_seq = 0
        self._rx_seq = 0
        self._header_formatter = struct.Struct('II')  # header format is (sequence number (UINT32), length (UINT32)).
        self._header_size = 8  # header size is 8 bytes

    @property
    def tx_seq(self):
        return self._tx_seq

    @property
    def rx_seq(self):
        return self._rx_seq

    def reset_seq(self):
        self._tx_seq = 0
        self._rx_seq = 0

    def send(self, msg_type, msg):
        data = {'type': msg_type, 'msg': msg}
        payload = json.dumps(data).encode(encoding='utf-8')
        header = self._header_formatter.pack(self._tx_seq, len(payload))
        self._sock.sendall(header + payload)
        self._tx_seq += 1

    def recv(self):
        header = self._sock.recv(self._header_size)
        seq, leng = self._header_formatter.unpack(header)
        if seq == self._rx_seq:
            self._rx_seq += 1
            payload = self._sock.recv(leng)
            data = json.loads(payload.decode(encoding='utf-8'))
            return data['type'], data['msg']
        else:
            return None
