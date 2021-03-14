import os
import sys
import pandas as pd
import numpy as np
import json

DATA_DIR = os.path.join(".","cs-train")

#for each json file:
  #read json into pd
df = pd.read_json(DATA_DIR+"/invoices-2017-11.json")

print(df.head())

 #check for inputs
 #correct them

 #concatenate into 1 dataframe

 #index on day

 #aggregate on price and country?
