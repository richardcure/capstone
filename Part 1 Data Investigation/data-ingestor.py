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

        #fill null columns:
        print('filling null values of price, stream_id, times_viewed columns')

        #price = total_price
        df['price'] = df['price'].fillna(df['total_price'])
        time.sleep(3)
        #stream_id = StreamID
        df['stream_id'] = df['stream_id'].fillna(df['StreamID'])
        time.sleep(3)
        #times_viewed = TimesViewed
        df['times_viewed'] = df['times_viewed'].fillna(df['TimesViewed'])
        time.sleep(3)

        print('null values filled, now checking this and dropping duplicate columns')

        #check for nulls and drop columns
        if (df['price'].isna().sum() == 0):
            print('price column has no nulls, dropping total_price column')
            df.drop(['total_price'], axis=1, inplace=True)
        if (df['stream_id'].isna().sum() == 0):
            print('stream_id column has no nulls, dropping StreamID column')
            df.drop(['StreamID'], axis=1, inplace=True)
        if (df['times_viewed'].isna().sum() == 0):
            print('times_viewed column has no nulls, dropping TimesViewed column')
            df.drop(['TimesViewed'], axis=1, inplace=True)

        #ouput head, tail and shape of concatenated data
        #print(df.head())
        #print(df.tail())
        #print(df.shape)

        #return as a dataframe
        return df

    except Exception as e:
        print('error reading json', e)

#call read function, passing in the list of JSON input files
df = read_json_input_data(file_list)

#export csv data
df.to_csv('transactions.csv')

print('csv data exported')
