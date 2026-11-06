"""test_threads - cora.set_num_threads and cora.num_threads.

Authors:       Tobias Ladner, Claude Sonnet 5.5
Written:       03-November-2026
Last update:   ---
Reviewed:      ---
"""
import unittest

import cora


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #


# ============================================  MAIN  ============================================ #

class ThreadsTest(unittest.TestCase):
    def tearDown(self):
        cora.set_num_threads(0)

    def test_set_and_reset(self):
        cora.set_num_threads(3)
        self.assertEqual(cora.num_threads(), 3)
        cora.set_num_threads(0)  # the default: the environment's count, else the machine's
        self.assertGreaterEqual(cora.num_threads(), 1)

    def test_pools_follow(self):
        if not cora.has_torch:
            self.skipTest("no torch")
        import torch
        cora.set_num_threads(2)
        self.assertEqual(torch.get_num_threads(), 2)


if __name__ == "__main__":
    unittest.main()

# ----------------------------------------  END OF CODE  ----------------------------------------- #
