import unittest
import numpy as np
import pandas as pd
from stock_prediction import get_data, preprocess_data, pred_split,train,test,predict
df = get_data()

s={'Adj. Close': [2.456],' HL_PCT':[22.38],'PCT_change' : [4.576], 'Adj. Volume':[1.245]}
pre= pd.DataFrame(s)

class Testpredict(unittest.TestCase):
    def test_predict(self):
        result=predict(pre)
        self.assertEqual(any(result),any([2.581755]))
