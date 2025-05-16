import unittest
import threading
from waitting_queue import ThreadSafeOrderedDict

class TestThreadSafeOrderedDict(unittest.TestCase):

    def test_basic_operations(self):
        tsod = ThreadSafeOrderedDict()
        tsod.set('a', 1)
        tsod.set('b', 2)
        self.assertEqual(tsod.get('a'), 1)
        self.assertEqual(tsod.get('b'), 2)
        self.assertIn('a', tsod)
        self.assertEqual(len(tsod), 2)

    def test_order_preservation(self):
        tsod = ThreadSafeOrderedDict()
        tsod.set('x', 10)
        tsod.set('y', 20)
        tsod.set('z', 30)
        self.assertEqual(tsod.pop_first(), ('x', 10))
        self.assertEqual(tsod.pop_first(), ('y', 20))

    def test_remove_and_get(self):
        tsod = ThreadSafeOrderedDict()
        tsod.set('key', 100)
        tsod.remove('key')
        self.assertIsNone(tsod.get('key'))
        self.assertNotIn('key', tsod)

    def test_thread_safety(self):
        tsod = ThreadSafeOrderedDict()

        def writer(start, end):
            for i in range(start, end):
                tsod.set(f'key{i}', i)

        threads = [
            threading.Thread(target=writer, args=(0, 1000)),
            threading.Thread(target=writer, args=(1000, 2000))
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(tsod), 2000)
        self.assertEqual(tsod.get('key0'), 0)
        self.assertEqual(tsod.get('key1999'), 1999)

    def test_concurrent_get_and_set(self):
        tsod = ThreadSafeOrderedDict()

        def setter():
            for i in range(1000):
                tsod.set(i, i * 2)

        def getter():
            for i in range(1000):
                _ = tsod.get(i)

        threads = [threading.Thread(target=setter), threading.Thread(target=getter)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Final check
        for i in range(1000):
            val = tsod.get(i)
            if val is not None:
                self.assertEqual(val, i * 2)

if __name__ == '__main__':
    unittest.main()