import unittest

def import_cosmo():
    try:
        from figaro.cosmology import CosmologicalParameters
        return 0
    except ImportError:
        return 1

class FIGAROtest(unittest.TestCase):
    def test_cosmology(self):
        self.assertEqual(import_cosmo(), 0, "LALsuite not installed properly. Please run 'conda install -c conda-forge lalsuite'")

if __name__ == '__main__':
    unittest.main()
