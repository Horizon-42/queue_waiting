from multiprocessing import Manager, Process, Lock

class ProcessSafeOrderedDict:
    def __init__(self):
        self.manager = Manager()
        self.dict = self.manager.dict()
        self.order = self.manager.list()  # 用于记录插入顺序
        self.lock = Lock()

    def set(self, key, value):
        with self.lock:
            if key not in self.dict:
                self.order.append(key)
            self.dict[key] = value

    def get(self, key):
        with self.lock:
            return self.dict.get(key)

    def pop_first(self):
        with self.lock:
            if self.order:
                key = self.order.pop(0)
                return (key, self.dict.pop(key, None))
            return None

    def remove(self, key):
        with self.lock:
            if key in self.dict:
                self.dict.pop(key)
                self.order.remove(key)

    def __setitem__(self, key, value):
        if key not in self.dict:
            self.order.append(key)
        self.dict[key] = value

    def __getitem__(self, key):
        return self.dict[key]

    def __delitem__(self, key):
        del self.dict[key]
        self.order.remove(key)

    def keys(self):
        with self.lock:
            return list(self.order)

    def __contains__(self, key):
        with self.lock:
            return key in self.dict

    def __len__(self):
        with self.lock:
            return len(self.dict)
