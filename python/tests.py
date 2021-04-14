########################################################################################################################
# IMPORTS

# modules
import unittest as ut

# local
from python import fb_robyn_func as frf


########################################################################################################################
# TESTS

class TestStuff(ut.TestCase):
    def test_fd_unit_format(self):
        """
        Test that formats come out as expected
        """

        x_in_test = 1545354165
        result = frf.unit_format(x_in=x_in_test)
        self.assertEqual(result, '1.5 bln')

        x_in_test = 654654
        result = frf.unit_format(x_in=x_in_test)
        self.assertEqual(result, '654.7 tsd')

        x_in_test = 984.654
        result = frf.unit_format(x_in=x_in_test)
        self.assertEqual(result, '985')


if __name__ == '__main__':
    ut.main()
