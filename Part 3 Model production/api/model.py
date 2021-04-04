import glob
import json
from fbprophet import Prophet
from fbprophet.serialize import model_to_json, model_from_json
import time,os,csv,sys,uuid
from datetime import date
import numpy as np
import pandas as pd
from logger import update_predict_log, update_train_log

## model specific variables (iterate the version and note with each change)
if not os.path.exists(os.path.join(".", "models")):
    os.mkdir("models")
if not os.path.exists(os.path.join(".", "data")):
    os.mkdir("data")

MODEL_DIR = os.path.join('.','models')
MODEL_VERSION = 0.1
MODEL_VERSION_NOTE = "FBProphet model for time-series"
DATA_DIR = os.path.join('.','data')
#list top 10 countries for filtering
top10countries = ['United Kingdom','EIRE','Germany','France','Norway','Spain','Hong Kong','Portugal','Singapore','Netherlands']


def read_json_input_data(file_list):
    """
    function to read in json data from a list of files and perform some data checking and manipulation
    """
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


def load_aavail_data():
    """
    function to load the training data from local storage and call the preprocess function
    """
    #setup working directory where data is stored and get JSON input files
    json_pattern = os.path.join(DATA_DIR,'*.json')
    file_list = glob.glob(json_pattern)
    print('file_list: ' + str(file_list))
    df = read_json_input_data(file_list)

    #remove
    print('calling preprocess')

    #preprocess
    df_preprocessed = preprocess(df)

    #remove
    print('data preprocessed')

    return(df_preprocessed)


def preprocess(df):
    """
    function to preprocess the training data from data ingestion
    create data columns, drop unneeded columns
    filter out top 10 df_individual_countries
    perform further columns manipulation
    return dataframe
    """
    #create date column
    #day, month and years to be concatenated into one value to be stored in this column
    df['date'] = np.NaN
    df['date'] = df['year'].astype(str) + df['month'].astype(str).str.zfill(2) + df['day'].astype(str).str.zfill(2)
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')

    #drop unneeded columns
    df.drop(['customer_id','invoice','stream_id','year', 'month', 'day','times_viewed'], axis=1, inplace=True)

    #list top 10 countries for filtering
    top10countries = ['United Kingdom','EIRE','Germany','France','Norway','Spain','Hong Kong','Portugal','Singapore','Netherlands']

    #filter out rows which are not in the top 10, save into new dataframe 'dftop10'
    top10indexes = df['country'].isin(top10countries)
    dftop10 = df.copy()
    dftop10 = dftop10[top10indexes]

    #make price NaN values 0
    dftop10['price'].fillna(0, inplace=True)

    #make date column datetime type and country column categorical
    dftop10['date'] = pd.to_datetime(dftop10['date'])
    dftop10['country'] = dftop10.country.astype('category')

    #rename columns for fbprophet requirements
    dftop10.columns = ['country','y','ds']

    print(dftop10.head())
    print(dftop10.tail())
    print(dftop10.shape)

    return dftop10

def split_preprocessed_df(dftop10, top10countries):
    """
    function to split dataframe for top 10 countries
    perform string manipulation on country name
    returns a dictionary of generated name + country's dataframe
    """
    #split dataframe into individual countries for training, save dataframes to values of a dictionary with keys 'df_<first 2 letters of the country>'
    #e.g. 'df_Un' for United Kingdom
    df_individual_countries = {}

    for c in top10countries:
        dfname = 'df_' + str(c)[:2]
        df_subsetted = dftop10[dftop10['country'] == c]
        df_individual_countries[dfname] = df_subsetted

    return df_individual_countries


def model_train():
    """
    function to train model
    call load aavail dataframe
    split loaded data by country
    train models and save to local storage
    log output
    """

    ## start timer for runtime
    time_start = time.time()

    #remove
    print('calling load aavail data')
    ## data ingestion
    df = load_aavail_data()

    #remove
    print('splitting dfs into individual country dfs')

    #split df into dfs per country
    df_individual_countries = split_preprocessed_df(df, top10countries)

    #remove
    print('now about to train models')

    #train and save model to models directory for each country
    for dfname, df in df_individual_countries.items():

        #remove
        print('to train: ' + str(dfname))
        print(df.head())
        print('NaNs:')
        print(df.isna().sum())
        model = Prophet()
        model.fit(df)

        modelname = 'model_' + str(dfname) + '.json'

        #remove
        print(str(modelname) + ' trained')

        with open(os.path.join((MODEL_DIR), modelname), 'w') as f:
            modeltosave = model_to_json(model)
            json.dump(modeltosave, f)
            print('saved ' + modelname)

    m, s = divmod(time.time()-time_start, 60)
    h, m = divmod(m, 60)
    runtime = "%03d:%02d:%02d"%(h, m, s)

    #remove
    print('updating train log')

    ## update the log file
    update_train_log(df['ds'].shape,
                    #eval_test,
                    runtime,
                    MODEL_VERSION,
                    MODEL_VERSION_NOTE,
                    )

    #remove
    print('updated train log')


def model_predict(country, year, month, day, model):
    """
    function to predict from model
    make a future dataframe for 30 days
    predict 30 days on the given model
    sum up predictions
    log output
    return the sum
    """

    ## start timer for runtime
    time_start = time.time()

    ## check date
    target_date = "{}-{}-{}".format(year,str(month).zfill(2),str(day).zfill(2))
    print(target_date)

    future = model.make_future_dataframe(periods=30)

    ## make prediction and gather data for log entry
    y_pred = model.predict(future.tail(30))
    y_pred_output = y_pred[['ds', 'yhat']]
    y_pred_sum = y_pred_output['yhat'].sum()
    print('predictions:')
    print(y_pred_output)
    print('30 day sum: ' + str(y_pred_sum))

    m, s = divmod(time.time()-time_start, 60)
    h, m = divmod(m, 60)
    runtime = "%03d:%02d:%02d"%(h, m, s)

    ## update the log file
    update_predict_log(country,y_pred_sum,target_date,runtime, MODEL_VERSION)

    return(y_pred_sum)


#loading model function
#for the given country
    #load its json model from project file
    #convert json to a Prophet model
    #return the model
def model_load(country):
    """
    function to load a model from local storage
    for the given country
        load its json model from project file
        convert json to a Prophet model
        return the model
    """

    countrydfnamemapping = {
        'United Kingdom' : 'df_Un',
        'EIRE' : 'df_EI',
        'Germany' : 'df_Ge',
        'France' : 'df_Fr',
        'Norway' : 'df_No',
        'Spain' : 'df_Sp',
        'Hong Kong' : 'df_Ho',
        'Portugal' : 'df_Po',
        'Singapore' : 'df_Si',
        'Netherlands' : 'df_Ne'
    }

    dfname = countrydfnamemapping[country]
    jsonmodelname = 'model_' + dfname + '.json'
    with open(os.path.join(MODEL_DIR, jsonmodelname), 'r') as f:
        jsonmodel = json.load(f)
        loadedmodel = model_from_json(jsonmodel)

    return loadedmodel


if __name__ == "__main__":

    """
    basic test procedure for model.py
    """

    ## train the models
    model_train()

    ## load the model
    model = model_load('Norway')

    ## example predict
    country='Norway',
    year='2019',
    month='08',
    day='01'

    result = model_predict(country, year, month, day, model)
    print('prediction: ' + str(result))
