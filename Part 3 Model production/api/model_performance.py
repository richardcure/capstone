import requests
import pandas as pd
import shutil, os
'''
for each future month
    add the month's data from /newdata to /data
    train on current data
    for each country
        predict current month
        write prediction to DataFrame


then:
create actuals from /newdata/ in json
merge predictions with actuals
generate visualisation of predictions vs actuals
'''

top10countries = ['United Kingdom','EIRE','Germany','France','Norway','Spain','Hong Kong','Portugal','Singapore','Netherlands']
future_months = ['2019-08', '2019-09', '2019-10', '2019-11', '2019-12']
headers = {'content-type': 'application/json'}
trainurl = 'http://localhost:8080/train'
predicturl = 'http://localhost:8080/predict'
newdatadir = './newdata/'
datadir = './data/'

#set up predictions dataframe
df = pd.DataFrame(columns=[top10countries])
df['year-month'] = future_months

for m in future_months:

    filetocopy = newdatadir + 'invoices-' + m + '.json'
    #print('filetocopy: '+filetocopy)
    shutil.copy(filetocopy, datadir)

    print('training for month ' + m)
    requests.get(trainurl, headers=headers)

    for c in top10countries:

        country = c
        year = str(m[:4])
        month = str(m[-2:])
        day = '01'
        print('country, year, month: ' + country + ' ' + year + ' ' + month)
        predictionjson = "{'country':'"+country+"','year':'"+year+"','month':'"+month+"','day':'"+day+"'}"
        #print('predictionjson:')
        #print(predictionjson)

        print('predicting for country ' + c)
        prediction = requests.post(predicturl, data=predictionjson, headers=headers)

        #TODO: write predictions to predictions dataframe

#TODO:
'''
create actuals from /newdata/ in json
merge predictions with actuals
generate visualisation of predictions vs actuals
'''
