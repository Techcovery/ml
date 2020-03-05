import unittest
import numpy as np
from stock_predict import get_data,split
df = get_data()

a = df.head(100)
b = a.iloc[:,-1]
class Testsplit(unittest.TestCase):
    def test_split(self):
        result = split(a,b)
        x = len(result[0])
        y = len(result[1])
        z = len(result[2])
        s = len(result[3])

        self.assertEqual(x,80)
        self.assertEqual(y,20)
        self.assertEqual(z,80)
        self.assertEqual(s,20)
