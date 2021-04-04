#!/usr/bin/env python
"""
module with functions to enable logging
"""

import time,os,re,csv,sys,uuid
from datetime import date

if not os.path.exists(os.path.join(".","logs")):
    os.mkdir("logs")

def update_train_log(x_shape, runtime, MODEL_VERSION, MODEL_VERSION_NOTE):
    """
    update train log file
    """

    ## name the logfile using something that cycles with date (day, month, year)
    today = date.today()
    logfile = os.path.join("logs", "train-{}-{}.log".format(today.year, today.month))

    ## write the data to a csv file
    header = ['unique_id','timestamp','x_shape', 'model_version','model_version_note','runtime']
    write_header = False
    if not os.path.exists(logfile):
        write_header = True
    with open(logfile, 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        if write_header:
            writer.writerow(header)

        to_write = map(str, [uuid.uuid4(), time.time(), x_shape, MODEL_VERSION, MODEL_VERSION_NOTE, runtime])
        writer.writerow(to_write)

def update_predict_log(country, y_pred, target_date, runtime, MODEL_VERSION):
    """
    update predict log file
    """

    ## name the logfile using something that cycles with date (day, month, year)
    today = date.today()
    logfile = os.path.join("logs", "predict-{}-{}.log".format(today.year, today.month))

    ## write the data to a csv file
    header = ['unique_id','timestamp',
    'country',
    'y_pred',
    'target_date',
    'runtime',
    'model_version']
    write_header = False
    if not os.path.exists(logfile):
        write_header = True
    with open(logfile,'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        if write_header:
            writer.writerow(header)

        to_write = map(str,[uuid.uuid4(), time.time(), country, y_pred, target_date, MODEL_VERSION, runtime])
        writer.writerow(to_write)

if __name__ == "__main__":

    """
    basic test procedure for logger.py
    """

    from model import MODEL_VERSION, MODEL_VERSION_NOTE

    ## train logger
    update_train_log('3000 X 1', 0.2, 0.1, 'test model version note')

    ## predict logger
    update_predict_log('United Kingdom', 'test predictions', '2019-08-01', 0.3, 0.1)
