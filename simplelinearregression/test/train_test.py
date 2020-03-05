import unittest
from height_weight import train

class TestTrain(unittest.TestCase):

    def test_train(self): 
        height=[[4.0],[4.5],[5.0],[5.2],[5.4],[5.8],[6.1],[6.2],[6.4],[6.8]]
        weight=[42,44,49,55,53,58,60,64,66,69]
        result = train(height,weight)
        self.assertEqual(result,(10.193621867881548, -0.4726651480637756))
        return result

if __name__ == '__main__':
    unittest.main()