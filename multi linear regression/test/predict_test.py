import unittest
import numpy as np
import pandas as pd
from stock_predict import get_data, preprocess_data, pred_split,train,test,predict
df = get_data()
#df = preprocess_data(df.head(100))
#X_pred,y_pred,X_testtrain,y_testtrain = pred_split(df.head(100))
s={'Adj. Close': [2.456],' HL_PCT':[22.38],'PCT_change' : [4.576], 'Adj. Volume':[1.245]}
pre= pd.DataFrame(s)
class Testpredict(unittest.TestCase):
    def test(self):
        pass
    def test_predict(self):
        #result=predict(pre)
        self.assertEqual(any(predict(pre)),any([2.581755]))
