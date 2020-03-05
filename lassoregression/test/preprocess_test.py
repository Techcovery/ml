import unittest
import pandas as pd
import numpy as np
from stock_predict import preprocess_data,get_data

s={'Open':[22.38,20.5],'High' : [np.nan,46.7], 'Low': [20.75,19.5], 'Close' : [20.75 ,'?'] , 'Volume' : [1225000 , 508900],'Adj. Open':[1.3364,1.25478] , 'Adj. High':[1.2547, 45.457] , 'Adj. Low':[4.2315,1.245],  'Adj. Close':[np.nan,1.245], 'Adj. Volume':[1.245,45.23] }
df= pd.DataFrame(s)

class Test_preprocess(unittest.TestCase):
    def test_preprocess(self):
        result=preprocess_data(df)
        self.assertEqual(result.iloc[0,0],-99999)


