# file: test_process_safe_ordereddict.py
import unittest
from multiprocessing import Process
from process_safe_queue import ProcessSafeOrderedDict

_shared_dict = None

def worker(start, end):
    global _shared_dict
    for i in range(start, end):
        _shared_dict.set(f"key{i}", i)

class TestProcessSafeOrderedDict(unittest.TestCase):

    def __init__(self, methodName="runTest"):
        global _shared_dict
        super().__init__(methodName)
        _shared_dict = ProcessSafeOrderedDict()

    def test_set_and_get(self):
        _shared_dict.set("a", 1)
        _shared_dict.set("b", 2)
        self.assertEqual(_shared_dict.get("a"), 1)
        self.assertEqual(_shared_dict.get("b"), 2)

    def test_overwrite(self):
        _shared_dict.set("a", 1)
        _shared_dict.set("a", 3)
        self.assertEqual(_shared_dict.get("a"), 3)

    def test_delete(self):
        _shared_dict.set("x", 9)
        _shared_dict.remove("x")
        self.assertNotIn("x", _shared_dict)
        self.assertEqual(None, _shared_dict.get("x"))

    def test_order(self):
        _shared_dict.clear()
        _shared_dict.set("one", 1)
        _shared_dict.set("two", 2)
        _shared_dict.set("three", 3)
        self.assertEqual(_shared_dict.keys(), ["one", "two", "three"])

    def test_len_and_contains(self):
        self.assertEqual(len(_shared_dict), 0)
        _shared_dict.set("a", 5)
        self.assertEqual(len(_shared_dict), 1)
        self.assertIn("a", _shared_dict)
        self.assertNotIn("b", _shared_dict)

    def test_process_safety(self):
        global _shared_dict

        # clear _shared_dict
        _shared_dict.clear()

        p1 = Process(target=worker, args=(0, 50))
        p2 = Process(target=worker, args=(50, 100))
        p1.start()
        p2.start()
        p1.join()
        p2.join()

        self.assertEqual(len(_shared_dict), 100)
        for i in range(100):
            self.assertEqual(_shared_dict.get(f"key{i}"), i)

if __name__ == "__main__":
    _shared_dict = ProcessSafeOrderedDict()
    worker(10, 20)
    # print(_shared_dict.keys())
    unittest.main()
