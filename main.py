
if __name__ == "__main__":
    # test_video("dataset/start.mp4")
    # test_images("dataset/images", (0, 10))
    import unittest
    from multiprocessing import Process
    from process_safe_queue import ProcessSafeOrderedDict

    global _shared_dict
    _shared_dict = ProcessSafeOrderedDict()

    class TestProcessSafeOrderedDict(unittest.TestCase):
        def __init__(self, methodName="runTest"):
            super().__init__(methodName)
            self._shared_dict = _shared_dict