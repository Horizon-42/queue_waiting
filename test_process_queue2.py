import unittest
from multiprocessing import Process
from time import sleep
from process_safe_queue import ProcessSafeOrderedDict
import multiprocessing

# def worker(d, prefix, count):
#     for i in range(count):
#         d[f"{prefix}_{i}"] = i
#         sleep(0.01)

class TestProcessSafeOrderedDict(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shared_dict = ProcessSafeOrderedDict()

    # def test_single_process_set_get(self):
    #     self.shared_dict['a'] = 1
    #     self.assertEqual(self.shared_dict['a'], 1)

    def test_multi_process_write(self):
        def worker(d, prefix, count):
            for i in range(count):
                key = f"{prefix}_{i}"
                d[key] = i
                sleep(0.01)  # 模拟并发

        # 清空旧数据（仅用于测试）
        for k in self.shared_dict.keys():
            del self.shared_dict[k]

        p1 = Process(target=worker, args=(self.shared_dict, 'p1', 5))
        p2 = Process(target=worker, args=(self.shared_dict, 'p2', 5))

        p1.start()
        p2.start()
        p1.join()
        p2.join()

        # 检查结果
        keys = self.shared_dict.keys()
        self.assertEqual(len(keys), 10)
        for i in range(5):
            self.assertIn(f'p1_{i}', keys)
            self.assertIn(f'p2_{i}', keys)

if __name__ == '__main__':
    multiprocessing.set_start_method('fork')

    unittest.main()
