import sys
sys.path.append(".")
import unittest
import numpy as np
from cgmcore.utils import subsample_pointcloud


subsampling_sizes = [5000, 10000, 20000, 30000, 40000, 50000]

#TODO fail some routines when calling with way to many points as requested oucome
#TODO test dimensions

class TestUtils(unittest.TestCase):

    def test_pcd_subsampling_random(self):
        """
        Tests random subsampling.
        """
        
        pointcloud_original = self.get_pointcloud()
        
        for subsampling_size in subsampling_sizes:
            pointcloud_subsampled = subsample_pointcloud(pointcloud_original, subsampling_size, "random")
            self.assertEqual(len(pointcloud_subsampled), subsampling_size)
        
        
    def test_pcd_subsampling_first(self):
        """
        Tests first subsampling.
        """
        
        pointcloud_original = self.get_pointcloud()
        
        
        # Generate some pointclouds and check if target_size is met.
        pointclouds = []
        for subsampling_size in subsampling_sizes:
            pointcloud_subsampled = subsample_pointcloud(pointcloud_original, subsampling_size, "first")
            self.assertEqual(len(pointcloud_subsampled), subsampling_size)
            pointclouds.append(pointcloud_subsampled)
            
        # See if the pointclouds have the same first n elements.
        for pointcloud1, pointcloud2 in zip(pointclouds[0:], pointclouds[1:]):
            min_length = min(len(pointcloud1), len(pointcloud2))
            self.assertTrue(min_length != 0)
            self.assertTrue(np.array_equal(pointcloud1[:min_length], pointcloud2[:min_length]))
    
    
    def test_pcd_subsampling_random(self):
        """
        Tests random sequential_skip.
        """
        
        pointcloud_original = self.get_pointcloud()
        
        for subsampling_size in subsampling_sizes:
            pointcloud_subsampled = subsample_pointcloud(pointcloud_original, subsampling_size, "sequential_skip")
            self.assertEqual(len(pointcloud_subsampled), subsampling_size)
    
    
    def get_pointcloud(self):
        return np.random.random((41232, 4))
    
#    def test_upper(self):
#        self.assertEqual('foo'.upper(), 'FOO')

#    def test_isupper(self):
#        self.assertTrue('FOO'.isupper())
#        self.assertFalse('Foo'.isupper())

#    def test_split(self):
#        s = 'hello world'
#        self.assertEqual(s.split(), ['hello', 'world'])
#        # check that s.split fails when the separator is not a string
#        with self.assertRaises(TypeError):
#            s.split(2)

if __name__ == '__main__':
    unittest.main()