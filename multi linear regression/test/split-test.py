import unittest
import numpy as np
from stock_predict import get_data,split
df = get_data()
#df = preprocess_data(df.head(100))
#X_pred,y_pred,X_testtrain,y_testtrain = pred_split(df.head(100))
#X_train, X_test, y_train, y_test = split(X_testtrain,y_testtrain)
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

if __name__ == '__main__':
    unittest.main()
