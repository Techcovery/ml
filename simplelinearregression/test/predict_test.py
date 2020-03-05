import unittest
from height_weight import pred 

class Testpred(unittest.TestCase):

    def test_pred(self): 
        height=[[5.5]]
        
        result = pred(height)
        #s = array([[55.59225513]
        self.assertEqual((result),([55.59225513]))
        return result

if __name__ == '__main__':
    unittest.main()