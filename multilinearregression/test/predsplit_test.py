import unittest
import numpy as np
from stock_predict import get_data, pred_split
df = get_data()

class Testpreprocess(unittest.TestCase):
    def test_pred_split(self):
        result = pred_split(df.head(100))

        x = len(result[0])
        y = len(result[1])
        z = len(result[2])
        s = len(result[3])

        self.assertEqual(x,1)
        self.assertEqual(y,1)
        self.assertEqual(z,99)
        self.assertEqual(s,99)
