import numpy as np
import os


class ItemDebugHandler(object):
    def __init__(self, path='./'):
        self.items = {}
        self.set_item_path(os.path.join(path, 'experiment_dataset.npy'))

    def set_item_path(self, path):
        self.path = path

    def add_item(self, item_name, item_value):
        if item_name not in self.items:
            self.items[item_name] = []
        self.items[item_name].append(item_value)

    def get_item(self, item_name):
        return np.array(self.items[item_name])

    def get_item_keys(self):
        return self.items.keys()

    def save_items(self):
        print('save ItemDebugHandler')
        np.save(self.path, self.items, allow_pickle=True)

    def load_items(self, path=None):
        if path is not None:
            self.items = np.load(path, allow_pickle=True).tolist()
        else:
            self.items = np.load(self.path, allow_pickle=True).tolist()


def test1():
    itemDebugHandler = ItemDebugHandler()
    itemDebugHandler.add_item('d1', 1)
    itemDebugHandler.add_item('d1', 1)
    itemDebugHandler.add_item('d1', 1)
    data = itemDebugHandler.get_item('d1')
    print(data)
    print(itemDebugHandler.get_item_keys())

    itemDebugHandler.add_item('d2', 2)
    itemDebugHandler.add_item('d2', 2)
    itemDebugHandler.add_item('d2', 2)
    data = itemDebugHandler.get_item('d2')
    print(data)
    print(itemDebugHandler.get_item_keys())

    itemDebugHandler.save_items()

    itemDebugHandler2 = ItemDebugHandler()
    itemDebugHandler2.load_items()
    print(itemDebugHandler2.items)


def test2():
    itemDebugHandler = ItemDebugHandler()
    itemDebugHandler.load_items()
    print(itemDebugHandler.items)


if __name__ == '__main__':
    test2()
