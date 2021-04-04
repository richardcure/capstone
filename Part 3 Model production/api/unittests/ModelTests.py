#!/usr/bin/env python
"""
model tests
"""

import sys, os
import unittest
sys.path.insert(1, os.path.join('..', os.getcwd()))

## import model specific functions and variables
from model import *




class ModelTest(unittest.TestCase):
    """
    test the essential functionality
    """

    def test_01_train(self):
        """
        test the train functionality
        """

        ## train the model
        model_train()
        self.assertTrue(os.path.exists(os.path.join("models", "model_df_Un.json")))

    def test_02_load(self):
        """
        test the train functionality
        """

        ## train the model
        model = model_load('EIRE')

        self.assertTrue('predict' in dir(model))
        self.assertTrue('fit' in dir(model))


    def test_03_predict(self):
        """
        test the predict function input
        """

        ## load model first
        model = model_load('France')

        ## ensure that a list can be passed
        country = 'France'
        year = '2019'
        month = '08'
        day = '01'

        result = model_predict(country, year, month, day, model)
        self.assertTrue(result)


### Run the tests
if __name__ == '__main__':
    unittest.main()
