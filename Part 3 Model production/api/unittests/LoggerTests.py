#!/usr/bin/env python
"""
model tests
"""

import os, sys
import csv
import unittest
from ast import literal_eval
import pandas as pd
import time,os,re,csv,sys,uuid,joblib
from datetime import date
sys.path.insert(1, os.path.join('..', os.getcwd()))

## import model specific functions and variables
from logger import update_train_log, update_predict_log



class LoggerTest(unittest.TestCase):
    """
    test the essential functionality
    """

    def test_01_train(self):
        """
        ensure log file is created
        """
        today = date.today()
        log_file = os.path.join("logs", "train-{}-{}.log".format(today.year, today.month))
        if os.path.exists(log_file):
            os.remove(log_file)

        ## update the log
        x_shape = (1000,)
        runtime = "00:00:01"
        model_version = 0.1
        model_version_note = "test model"

        update_train_log(x_shape, runtime, model_version, model_version_note)

        self.assertTrue(os.path.exists(log_file))

    def test_02_train(self):
        """
        ensure that content can be retrieved from log file
        """
        today = date.today()
        log_file = os.path.join("logs", "train-{}-{}.log".format(today.year, today.month))

        ## update the log
        x_shape = (2000,)
        runtime = "00:00:02"
        model_version = 0.1
        model_version_note = "test model"

        update_train_log(x_shape, runtime, model_version, model_version_note)

        df = pd.read_csv(log_file)
        logged_x_shape = [literal_eval(i) for i in df['x_shape'].copy()][-1]
        self.assertEqual(x_shape, logged_x_shape)


    def test_03_predict(self):
        """
        ensure log file is created
        """
        today = date.today()
        log_file = os.path.join("logs", "predict-{}-{}.log".format(today.year, today.month))
        if os.path.exists(log_file):
            os.remove(log_file)

        ## update the log
        country = 'United Kingdom'
        y_pred = 30000
        runtime = "00:00:03"
        model_version = 0.1
        target_date = "('2019',)-('08',)-01"

        update_predict_log(country, y_pred, target_date, runtime, model_version)

        self.assertTrue(os.path.exists(log_file))


    def test_04_predict(self):
        """
        ensure that content can be retrieved from log file
        """
        today = date.today()
        log_file = os.path.join("logs", "predict-{}-{}.log".format(today.year, today.month))

        ## update the log
        country = 'United Kingdom'
        y_pred = 40000
        runtime = "00:00:04"
        model_version = 0.1
        target_date = "('2019',)-('08',)-01"

        update_predict_log(country, y_pred, target_date, runtime, model_version)

        df = pd.read_csv(log_file)
        logged_y_pred = [literal_eval(str(i)) for i in df['y_pred'].copy()][-1]
        self.assertEqual(y_pred, logged_y_pred)


### Run the tests
if __name__ == '__main__':
    unittest.main()
