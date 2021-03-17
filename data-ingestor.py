import os
import sys
import pandas as pd
import numpy as np
import json
import glob
import time


#setup working directory where data is stored and get JSON input files
DATA_DIR = os.path.join('.','cs-train')
json_pattern = os.path.join(DATA_DIR,'*.json')
file_list = glob.glob(json_pattern)
print('file_list: ' + str(file_list))


def read_json_input_data(file_list):

    #create an empty list to store the data frames
    dfs = []

    try:
        #for each json file:
        for file in file_list:
            #read the json
            print('reading file')
            data = pd.read_json(file, orient='columns')
            print('read file')
            #append data frame to dfs
            dfs.append(data)
            print('appended data to dfs')

        #concatenate all the data frames into one df
        df = pd.concat(dfs)

        print(df.columns.to_list())

        #fill null columns:
        print('filling null values of price, stream_id, times_viewed columns')

        #price = total_price
        df['price'].fillna(df['total_price'])
        #stream_id = StreamID
        df = df['stream_id'].fillna(df['StreamID'])
        #times_viewed = TimesViewed
        df = df['times_viewed'].fillna(df['TimesViewed'])

        #check for nulls and drop columns
        print(df['price'].isna().sum())

        if (df['price'].isna().sum() == 0):
            print('price column has no nulls, dropping total_price column')
            df = df['total_price'].drop()

        if (df['stream_id'].isna().sum() == 0):
            print('stream_id column has no nulls, dropping StreamID column')
            df = df['StreamID'].drop()
        if (df['times_viewed'].isna().sum() == 0):
            print('times_viewed column has no nulls, dropping TimesViewed column')
            df = df['TimesViewed'].drop()

        #ouput head, tail and shape of concatenated data
        #print(df.head())
        #print(df.tail())
        #print(df.shape)

        #return as a dataframe
        return df

    except Error as e:
        print('error reading json', e)

#call read function, passing in the list of JSON input files
df = read_json_input_data(file_list)

#export csv data
#df.to_csv('transactions.csv')

print('csv data exported')
