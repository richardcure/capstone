import os
import sys
import pandas as pd
import numpy as np
import json
import glob


#setup working directory where data is stored and get JSON input files
DATA_DIR = os.path.join(".","cs-train")
json_pattern = os.path.join(DATA_DIR,'*.json')
file_list = glob.glob(json_pattern)
print("file_list: " + str(file_list))


def read_json_input_data(file_list):

    #create an empty list to store the data frames
    dfs = []

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

    #ouput head, tail and shape of concatenated data
    print(df.head())
    print(df.tail())
    print(df.shape)

    #return as a dataframe
    return df

#call read function, passing in the list of JSON input files
df = read_json_input_data(file_list)

#export csv data
df.to_csv('transactions_raw.csv')

print('csv data exported')
