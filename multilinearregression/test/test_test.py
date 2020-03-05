import unittest
import numpy as np
from stock_predict import get_data, preprocess_data, pred_split,train,test,split
df = get_data()

X=df.head(100)
y=X.iloc[:,-1]


class Testtest(unittest.TestCase):
    def test_test(self):
        result=test(X,y)
        self.assertEqual(result,0.9636357115699786)
        