# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 15:55:21 2025

@author: Robin Buijs
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from datetime import datetime, timedelta
from shapely.geometry import Point, LineString, MultiLineString
import seaborn as sns
import matplotlib.pyplot as plt
import locale
from io import StringIO

API_token = '7232a6a9-896c-42a1-8292-33821af5925e'


    
def find_prices(row, prices):
    prices_filter = prices.loc[prices['datetime_CET_end'] > row['Begindatum en -tijd']]
    prices_filter = prices_filter.loc[prices_filter['datetime_CET'] < row['Einddatum en -tijd']]

    result = prices_filter.assign(**row.to_dict())
    return result

def create_date_range(row):
    result = pd.date_range(start=row['Begindatum'], end=row['Einddatum'], freq='D')
    
    return result

# Vind alle datums in de dataset die bij rustactiviteiten horen
def unique_dates(df):
    df1 = df.copy()
    df1 = df1[df1['Laden']==1]
    
    df1['Begindatum'] = df1['Begindatum en -tijd'].dt.date
    df1['Einddatum'] = df1['Einddatum en -tijd'].dt.date
    df1['daterange'] = df1.apply(create_date_range, axis=1)
    df1 = df1.explode('daterange')
    datums_uniek = df1['daterange'].drop_duplicates()
    return datums_uniek

locale.setlocale(locale.LC_TIME, 'nl_NL.utf8')

df = pd.read_excel('template.xlsx', sheet_name = 'ritten')
df['Positie'] = df['Positie'].str.strip()

csv = """
Activiteit
Rusten
"""

df_act = pd.read_csv(StringIO(csv)).assign(Laden = 1)

df['Begindatum en -tijd'] = pd.to_datetime(df['Begindatum en -tijd'])
df['Einddatum en -tijd'] = pd.to_datetime(df['Einddatum en -tijd'])
df = df.sort_values(['Voertuig', 'Begindatum en -tijd']).reset_index(drop = True)

df['Datum'] = pd.to_datetime(df['Begindatum en -tijd']).apply(lambda x: x.date())

df = df.merge(df_act, how = 'left', on = 'Activiteit')
df['Laden'] = df.Laden.fillna(0)

datums_uniek = unique_dates(df)

prices = pd.read_excel('day_ahead_prices.xlsx')
#prices['datetime_CET'] = pd.to_datetime(prices['datetime_CET'].astype('datetime64[ns]')) 

#a = get_day_ahead_prices(API_token, '20251009')

variable_price = True

if variable_price:
    for i in datums_uniek:
        i = pd.Timestamp(i).tz_localize(None)
        if not (prices['datetime_CET'].dt.date == i.date()).any():
            print(f"Prijzen voor {i} ontbreken — ophalen via ENTSO-E API")
            # Probeer prijzen voor datum i op te halen op ENTSO-E API
            try:
                a = get_day_ahead_prices(API_token, i.strftime('%Y%m%d'))
                a['datetime_CET'] = a['datetime_CET'].dt.tz_localize(None)
                a['datetime_UTC'] = a['datetime_UTC'].dt.tz_localize(None)
                print(f'Prices found for {i}')
                
                # Voeg prijsdata samen met bestaande prijsdata
                prices = pd.concat([prices,a])
            # Wanneer API fout geeft, throw exception
            except:
                print(f'No day ahead prices available for {i}')

else:
    # Werk met een vaste elektriciteitsprijs, gegeven door het bedrijf of een standaardwaarde
    print('fixed_prices')
    
prices = prices[['datetime_CET', 'price_eur_mwh']].sort_values(by = 'datetime_CET')
prices['datetime_CET_end'] = prices['datetime_CET'].shift(-1)
prices['Datum'] = prices['datetime_CET'].dt.date

df_laden = df[df['Laden']==1]
df_laden = pd.concat((find_prices(row, prices) for _, row in df_laden.iterrows()), ignore_index=True)
df_laden['Begindatum en -tijd'] = df_laden[['Begindatum en -tijd', 'datetime_CET']].max(axis = 1)
df_laden['Einddatum en -tijd'] = df_laden[['Einddatum en -tijd', 'datetime_CET_end']].min(axis = 1)
df_laden = df_laden[['Voertuig', 'Begindatum en -tijd', 'Einddatum en -tijd', 'Positie', 'Afstand', 'Activiteit', 'Datum', 'Laden', 'price_eur_mwh']]

df = df.loc[df['Laden']==0]
df = pd.concat([df, df_laden])


df['Duur'] = (df['Einddatum en -tijd'] - df['Begindatum en -tijd']).apply(lambda x: x.total_seconds())
df['nacht'] = np.where(((df.Afstand < 3) & (df.Duur > 6*3600)),1,0)

csv2 = """
Positie
thuisbasis
"""

df_locatie = pd.read_csv(StringIO(csv2)).assign(thuis = 1)

df = df.merge(df_locatie, how = 'left', on = 'Positie')
df['thuis'] = df.thuis.fillna(0)

nachtladen = 1
activiteitenladen = 1

if (nachtladen == 0) & (activiteitenladen == 0):
    df['Laadtijd'] = np.where((d2['thuis'] == 1), df['Duur'],0) # alleen thuis laden
elif (nachtladen == 1) & (activiteitenladen == 0):
    df['Laadtijd'] = np.where((df['thuis'] == 1) | (df['nacht'] == 1), df['Duur'],0) # thuis of 's nachts laden
elif (nachtladen == 0) & (activiteitenladen == 1):
    df['Laadtijd'] = np.where((df['thuis'] == 1) | (df['Laden'] == 1), df['Duur'],0) # thuis of tijdens activiteit
elif (nachtladen == 1) & (activiteitenladen == 1):
    df['Laadtijd'] = np.where((df['thuis'] == 1) | (df['Laden'] == 1) | (df['nacht'] == 1), df['Duur'],0) # thuis, 's nachts of tijdens activiteit

df['Laadtijd'] = np.where((df['Activiteit'] == 'Rijden') | (df['Afstand'] >= 3), 0, df['Laadtijd']) # niet AC-laden tijdens rijden  		
df = df.sort_values(['Voertuig', 'Begindatum en -tijd']).reset_index()


# Function to connect with ENTSO-E API and to retrieve 

# token: API_token: first request an API-token to the ENTSO-E transparency platform
# date_start: datetime as %Y%m%d
# date_end: leave empty if request is for one day only


def get_day_ahead_prices(token, date_start, date_end = ''):
    
    import requests
    import pandas as pd
    from datetime import datetime, timedelta
    import xmltodict
    import json
    from zoneinfo import ZoneInfo

    # Set end date to start date if request is for one day only
    if date_end == '':
        date_end = date_start

    date_range = pd.date_range(
        start=pd.to_datetime(date_start).date(),
        end=pd.to_datetime(date_end).date(),
    )

    df = pd.DataFrame()

    # Iterate over dates
    for i in date_range:
        start = i.strftime('%Y%m%d0000')
        end = i.strftime('%Y%m%d2359')
    
        # ENTSO-E parameters
        params = {
            'securityToken': token,
            'documentType': 'A44',  # Day-ahead prices
            'in_Domain': '10YNL----------L', # Bidding zone NL
            'out_Domain': '10YNL----------L',
            'periodStart': start,
            'periodEnd': end,
        }
        
        # API request
        url = 'https://web-api.tp.entsoe.eu/api'
        response = requests.get(url, params=params)
        
        if response.status_code != 200:
            raise Exception(f"Fout bij API-aanroep: {response.status_code} - {response.text}")
    
        # Unpack dictionary of response
        o = xmltodict.parse(response.text)
        list_resp = o['Publication_MarketDocument']['TimeSeries'][0]
        
        dict_values = list_resp['Period']['Point']
        
        # Create DataFrame from response
        df_temp = pd.DataFrame()
        for i in range(len(dict_values)):
            df_temp = pd.concat([df_temp, pd.DataFrame.from_dict([dict_values[i]])], ignore_index=True)

        # Add UTC date to dataframe
        date_UTC = datetime.strptime(list_resp['Period']['timeInterval']['start'], "%Y-%m-%dT%H:%MZ")
        date_UTC = date_UTC.replace(tzinfo = ZoneInfo('UTC'))
        
        df_temp['date_UTC'] = date_UTC

        if len(df_temp) < 30:
            frequency = 'h'
            div = 1
        else:
            frequency = '15min'
            div = 4
        df = pd.concat([df, df_temp])

    df['position'] = df['position'].astype(int) - 1
    df['price.amount'] = df['price.amount'].astype(float)

    # Calculate UTC datetime
    df = df.assign(datetime_UTC = lambda d: d['date_UTC'] + pd.to_timedelta(d['position'], unit = 'h')/div)
    # Interpolation of datetime_UTC, to find missing hours
    df = df.set_index('datetime_UTC')
    full_index = pd.date_range(df.index.min(), df.index.max(), freq=frequency)
    df = df.reindex(full_index).reset_index()

    # Convert to CET time, keep necessary columns
    df['datetime_CET'] = df['index'].dt.tz_convert('Europe/Amsterdam')
    df = df.drop(columns=['position', 'date_UTC'])
    df = df.rename(columns = {'index': 'datetime_UTC', 'price.amount': 'price_eur_mwh'})
        
    df = df.iloc[:, [0, 2, 1]]
    
    return df
