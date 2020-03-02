import unittest
import numpy as np
from stock_predict import get_data, preprocess_data, pred_split,train
df = get_data()
#it gives model as  output
x=df.head(100)
y=x.iloc[:,-1]
class Testtrain(unittest.TestCase):
    def test_train(self):
        result=train(x,y)
        m = result[0]
        b = result[1]
        print(m)
        print(b)
        self.assertEqual(all(m),all( [ 9.91819922e-01, 6.50503927e-03,-2.52210730e-03,-3.52115871e-09]))
        self.assertEqual(b,0.011803660202904798)
        return result

if __name__ == '__main__':
    unittest.main()
