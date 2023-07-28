import numpy as np
import torch


class MemoryTerm(object):
    def __init__(self, MAX_MEMORY_SIZE, SEQUENCE_LENGTH, STEP_DATA_SHAPE, DEVICE=None, userDefinedSettings=None, is_lstm_hidden=False):
        if userDefinedSettings is not None:
            self.userDefinedSettings = userDefinedSettings
            self.DEVICE = self.userDefinedSettings.DEVICE
        else:
            self.DEVICE = DEVICE
        self.MAX_MEMORY_SIZE = MAX_MEMORY_SIZE
        self.SEQUENCE_LENGTH = SEQUENCE_LENGTH
        self.buffer = CircularQueue(MAX_MEMORY_SIZE=MAX_MEMORY_SIZE, DATA_SHAPE=[SEQUENCE_LENGTH, *STEP_DATA_SHAPE], is_lstm_hidden=is_lstm_hidden)
        self.episode_data = []

    def clear(self):
        self.buffer.clear_queue()

    def push(self, data, current_buffer_index):
        self.episode_data.append(data)
        if len(self.episode_data) >= self.SEQUENCE_LENGTH:
            assert len(self.episode_data) == current_buffer_index + 1, 'pushing episode data is shifted'
            self.push_episode()
            self.episode_memory_reset()

    def push_episode(self):
        self.buffer.append(self.episode_data)

    def episode_memory_reset(self):
        self.episode_data.clear()

    def sample(self, batch_size=None, sampling_method='random', index=None):
        state_sequence_batch = torch.FloatTensor(self.buffer.get(batch_size=batch_size, how=sampling_method, index=index)).to(self.DEVICE)
        return state_sequence_batch

    def __len__(self):
        return len(self.buffer)


class CircularQueue(object):
    def __init__(self, MAX_MEMORY_SIZE, DATA_SHAPE, dtype=np.float32, is_lstm_hidden=False):
        self.MAX_MEMORY_SIZE = MAX_MEMORY_SIZE
        self.DATA_SHAPE = DATA_SHAPE
        self.dtype = dtype
        self.is_lstm_hidden = is_lstm_hidden
        self.clear_queue()

    def clear_queue(self):
        self.circular_queue = np.empty((self.MAX_MEMORY_SIZE, *self.DATA_SHAPE), dtype=self.dtype)
        self.current_queue_index = 0
        self.current_queue_size = 0

    def append(self, data):
        if self.is_lstm_hidden:
            formated_data = data[0].cpu().detach().numpy().reshape(1, -1)
        else:
            formated_data = np.array(data, dtype=self.dtype)
            if len(formated_data.shape) == 1:
                formated_data = formated_data.reshape(-1, 1)
        self.circular_queue[self.current_queue_index] = formated_data
        self.set_next_queue_index()

    def set_next_queue_index(self):
        if self.current_queue_size == self.MAX_MEMORY_SIZE - 1:
            self.current_queue_index = 0
            self.current_queue_size += 1
        elif self.current_queue_size < self.MAX_MEMORY_SIZE - 1:
            self.current_queue_index += 1
            self.current_queue_size += 1
        elif self.current_queue_index >= self.MAX_MEMORY_SIZE - 1:
            self.current_queue_index = 0
        else:
            self.current_queue_index += 1

    def get(self, batch_size, how='random', index=None):
        if index is not None:
            data = self.circular_queue[index]
        elif how == 'random':
            assert batch_size <= self.current_queue_index, 'choose index within current sample number'
            data = self.get_random(batch_size)
        elif how == 'last':
            assert batch_size <= self.current_queue_index, 'choose index within current sample number'
            data = self.get_last(batch_size)
        else:
            assert False, 'choose correct sampling method of replay memory'

        if self.is_lstm_hidden:
            sequence_index = 1
            data = data.squeeze(sequence_index)
        return data

    def get_random(self, size):
        index = np.random.randint(low=0, high=self.current_queue_index, size=size)
        return self.circular_queue[index]

    def get_last(self, size):
        if self.current_queue_index - size < 0:
            remain = size - self.current_queue_index + 1
            return np.concatenate([self.circular_queue[:self.current_queue_index], self.circular_queue[-remain + 1:]])
        else:
            return self.circular_queue[-size:]

    def __len__(self):
        return self.current_queue_size
