import hack1
import unittest

class TestBasicFunc(unittest.TestCase):

    def test_a_plus_b(self):
        res = hack1.add(1,2)
        return self.assertEqual(res,3)
    
if __name__ == '__main__':
   unittest.main()