# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 09:23:51 2018

@author: kazin
"""

import pyodbc as cn
import numpy as np
import seaborn as sns
import pandas as pd
import scipy.stats.stats as sp
from sklearn import preprocessing

#Connecting to database
server = 'facil.database.windows.net'
database = 'main'
username = 'facildatabase'
password = 'DifficultPassword69.'
driver = '{ODBC Driver 11 for SQL Server}'
cnxn = cn.connect('DRIVER=' + driver + ';SERVER=' + server + ';DATABASE=' + database + ';UID=' + username + ';PWD=' + password)
print(cnxn)

##Importing Data to dataframes form daatabase
nutsComplete = pd.read_sql('SELECT * from dbo.NutsComplete', con = cnxn)
boltsComplete = pd.read_sql('SELECT * from dbo.BoltsComplete', con = cnxn)
contracts = pd.read_sql('SELECT * from dbo.contracts', con = cnxn)
coatings = pd.read_sql('SELECT * from dbo.coatings', con = cnxn)
geometryCoating = pd.read_sql('SELECT * from dbo.geometryCoating', con = cnxn)
countryCodes = pd.read_sql('SELECT * from dbo.countryCodes', con = cnxn)
rfqPortal = pd.read_sql('SELECT * from dbo.rfqPortal', con = cnxn)

###Exporting dataframes to .csv files
#nutsComplete.to_csv('nutsComplete.csv', sep='\t', encoding='utf-8')
#boltsComplete.to_csv('boltsComplete.csv', sep= '\t', encoding='utf-8')
#contracts.to_csv('contracts.csv', sep= '\t', encoding='utf-8')
#coatings.to_csv('coatings.csv', sep= '\t', encoding='utf-8')
#geometryCoating.to_csv('geometryCoating.csv', sep= '\t', encoding='utf-8')
#countryCodes.to_csv('countryCodes.csv', sep= '\t', encoding='utf-8')

    
#KPI Calculation for boltsComplete
headers = list(boltsComplete)
KpiBolts =[]
boltsComplete = boltsComplete.astype('float64')

for x in range(len(headers)):
          KpiBolts.append((boltsComplete['Pprice'].corr(boltsComplete[headers[x]].astype(float))))
          
labels = ['ColumnName','Value']
KPIBolts = pd.DataFrame(np.column_stack([headers, KpiBolts]),
                    columns=['ColumnName','Value'])
##Exporting the KPI calculation boltsComplete  values into .csv
KPIBolts.to_csv('KpiBolts.csv',sep=',',encoding='utf-8')
#Replacing all the nan values with 0
boltsComplete.fillna(0, inplace = True)

##Normalizing the data
scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
scaled = scaler.fit_transform(boltsComplete)
BoltsNormalizedDataFrame = pd.DataFrame(scaled,columns=headers)

#KPI Calculation for nutsComplete
headers = list(nutsComplete)
KpiNuts =[]
nutsComplete['Has_Washer'] = nutsComplete['Has_Washer'].astype(int)
nutsComplete = nutsComplete.astype('float64')
for x in range(len(headers)):
          KpiNuts.append((nutsComplete['Pprice'].corr(nutsComplete[headers[x]].astype(float))))

labels = ['ColumnName','Value']
KPIN = pd.DataFrame(np.column_stack([headers, KpiNuts]),
                    columns=['ColumnName','Value'])
##Exporting the KPI calculation nutsComplete values into .csv
KPIN.to_csv('KpiNuts.csv',sep=',',encoding='utf-8')

#Replacing all the nan values with 0
nutsComplete.fillna(0, inplace = True)
scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
scaled = scaler.fit_transform(nutsComplete)
NutsNormalizedDataFrame = pd.DataFrame(scaled, columns = headers)

#Creating matrix for most valuables KPI in Bolts DataFrame
HeadDiameter_matrix = BoltsNormalizedDataFrame['HeadDiameter'].as_matrix()
Length_matrix = BoltsNormalizedDataFrame['Length'].as_matrix()
TotalLength_matrix = BoltsNormalizedDataFrame['TotalLength'].as_matrix()
ShoulderDiameter_matrix = BoltsNormalizedDataFrame['ShoulderDiameter'].as_matrix()
Weight_matrix = BoltsNormalizedDataFrame['Weight'].as_matrix()
TEGWNE_matrix = BoltsNormalizedDataFrame['TEGWNE'].as_matrix()

#Creating matrix for most valuables KPI in Nuts DataFrame
Diameter_matrix = NutsNormalizedDataFrame['Diameter'].as_matrix()
Height_matrix = NutsNormalizedDataFrame['Height'].as_matrix()
TEGWNE_Nuts_matrix = NutsNormalizedDataFrame['TEGWNE'].as_matrix()




  