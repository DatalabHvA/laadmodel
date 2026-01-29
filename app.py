import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from  matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import requests
from io import BytesIO
from datetime import timedelta
from datetime import datetime
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from itertools import product
import base64

@st.cache_data
# TODO: Functie mogelijk overbodig. Verwijderen in dat geval
def tekort_snel(df2, battery = 540, zuinig = 1.26):
    verbruik_rit = df2.Afstand.sum()*zuinig

    df2 = df2.sort_values('Begindatum en -tijd', ascending = False)
    tekort2 = []
    tekort1 = []

    for i, row in df2.reset_index().iterrows():
        tekort = min(
                        min(df2.iloc[i].laad_potentiaal1,
                            max(0, battery + df2.iloc[0:i+1].Afstand.sum()*zuinig - sum(tekort1) - sum(tekort2))
                         ), 
                        max(0, verbruik_rit - sum(tekort1) - sum(tekort2))
                     )

        tekort1.append(tekort)
        
        tekort_snel = max(0,df2.iloc[0:i+1].Afstand.sum()*zuinig - sum(tekort1) - sum(tekort2))
        tekort2.append(tekort_snel)
    return_df = pd.DataFrame({'bijladen_snel' : tekort2, 'bijladen' : tekort1}, index = df2.index)
    return return_df

@st.cache_data
# Hulpfunctie voor het vinden van alle datums bij laadactiviteiten
def create_date_range(row):
    result = pd.date_range(start=row['Begindatum'], end=row['Einddatum'], freq='D')
    return result

@st.cache_data
# Vind alle datums in de dataset die bij laadactiviteiten horen
def unique_dates(df):
    df1 = df.copy()
    df1 = df1[df1['Laden']==1]
    if df1.empty:
        datums_uniek = pd.Series(min(df['Begindatum en -tijd'].dt.date))
    else:
        df1['Begindatum'] = df1['Begindatum en -tijd'].dt.date
        df1['Einddatum'] = df1['Einddatum en -tijd'].dt.date
        df1['daterange'] = df1.apply(create_date_range, axis=1)
        df1 = df1.explode('daterange')
        datums_uniek = df1['daterange'].drop_duplicates()
    return datums_uniek

@st.cache_data
# Match prijzen met de juiste rijen in de data
def match_prices(row, prices):
    
    prices_filter = prices.loc[prices['datetime_CET_end'] > row['Begindatum en -tijd']]
    prices_filter = prices_filter.loc[prices_filter['datetime_CET'] < row['Einddatum en -tijd']]

    result = prices_filter.assign(**row.to_dict())
    
    # TODO: zorg ervoor dat tijdstippen waar geen prijs voor bestaat niet verwijderd worden. Oplossing: Plak nullen op einde, tussenliggende punten worden al doorgetrokken
    return result

@st.cache_data
# Functie voor het koppelen van prijzen aan het df
def add_day_ahead_prices(df, prices, aansluittijd = [600]):    
    
    #type_voertuig = int(np.minimum(df['Type voertuig'].min(), type_voertuigen)) - 1
    #aansluittijd = aansluittijd[type_voertuig]
        
    df_laden = df[df['Laden']==1]
    if df_laden.empty:
        df['Aansluittijd'] = np.nan
        df['price_eur_mwh'] = np.nan
        result = df
    else:
        df_laden = pd.concat((match_prices(row, prices) for _, row in df_laden.iterrows()), ignore_index=True)
        df_laden['Aansluittijd'] = df_laden.apply(lambda g: min(aansluittijd[g['Type voertuig']-1], (g['datetime_CET_end'] - g['Begindatum en -tijd']).total_seconds(), max(aansluittijd[g['Type voertuig']-1] - (g['datetime_CET'] - g['Begindatum en -tijd']).total_seconds(), 0)), axis = 1)
        df_laden['Begindatum en -tijd'] = df_laden[['Begindatum en -tijd', 'datetime_CET']].max(axis = 1)
        df_laden['Einddatum en -tijd'] = df_laden[['Einddatum en -tijd', 'datetime_CET_end']].min(axis = 1)
        df_laden = df_laden[['Voertuig', 'Type voertuig', 'Activiteit_id', 'Begindatum en -tijd', 'Einddatum en -tijd', 'Positie', 'Afstand', 'Activiteit', 'Datum', 'nacht', 'Laden', 'Aansluittijd', 'price_eur_mwh']]
        df_niet_laden = df[df['Laden']!=1]
        df_niet_laden['Aansluittijd'] = np.nan
        result =  pd.concat([df_niet_laden, df_laden])
        
    return result
        
@st.cache_data
# Functie voor het ophalen van extra data van de ENTSO-E API. Vraag hiervoor een API-token op   
def get_day_ahead_prices(date_start, date_end = ''):
    
    import requests
    import pandas as pd
    from datetime import datetime, timedelta
    import xmltodict
    import json
    from zoneinfo import ZoneInfo
    import streamlit as st
    
    API_token = st.secrets["api_keys"]["api_entsoe"]

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
            'securityToken': API_token,
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

# Functie om de geaggregeerde kosten voor laden voor een gehele activiteit te berekenen
@st.cache_data
def aggregate_hourly_costs(df):
    other_columns = df.columns.difference(['Activiteit', 'Positie', 'Afstand', 'Begindatum en -tijd', 'Einddatum en -tijd', 'bijladen', 'bijladen_snel', 'Duur'])
    other_dict = {key : 'first' for key in other_columns}

    agg_dict = {'Activiteit' : 'first',
                'Positie' : 'first',
                'Afstand' : 'first',
                'Begindatum en -tijd' : 'min',
                'Einddatum en -tijd' : 'max',
                'bijladen': 'sum',
                'bijladen_snel': 'sum',
                'Duur': 'sum'}
    
    agg_dict = {**agg_dict, **other_dict}
    # Bereken los de totale laadkosten en gemiddelde prijs per kwh voor de laadregels
    laadkosten = df.groupby('Activiteit_id').apply(lambda g: np.dot(g['price_eur_mwh'], g['bijladen'])/1000).rename('Laadkosten (EUR)')
    gemiddelde_prijs = df.groupby('Activiteit_id').apply(lambda g: np.dot(g['price_eur_mwh'], g['bijladen'])/(1000*g['bijladen'].sum())).rename('Gemiddelde laadprijs (EUR/kWh)')

    df_g = df.groupby('Activiteit_id').agg(agg_dict)
    df = df_g.join([laadkosten, gemiddelde_prijs])

    df['Laadkosten (EUR)'] = df['Laadkosten (EUR)'].fillna(0)
    df['Gemiddelde laadprijs (EUR/kWh)'] = df['Gemiddelde laadprijs (EUR/kWh)'].fillna(0)

    return df

@st.cache_data
# Functie voor het toevoegen van een extra regel als aan het einde van de rit de accu nog niet terug is volgeladen
def bijladen_einde_rit(df, prices, laadvermogen = [44], battery = [540], aansluittijd = [600], type_voertuigen = 1):
    type_voertuig = int(np.minimum(df['Type voertuig'].min(), type_voertuigen)) - 1

    battery = battery[type_voertuig]
    laadvermogen = laadvermogen[type_voertuig]
    aansluittijd = aansluittijd[type_voertuig]
    
    df_result = df.copy()
   
    df = df.fillna(method = 'ffill')
    lastrow = df.iloc[-1].copy()
    eindstand = lastrow['energie'] + lastrow['verbruik'] + lastrow['bijladen'] + lastrow['bijladen_snel']

    if eindstand < battery:
        # Modify fields in the duplicated row as needed:
        lastrow['Begindatum en -tijd'] = lastrow['Einddatum en -tijd']
        lastrow['Afstand'] = 0
        lastrow['Positie'] = 'Einde rit'
        lastrow['Activiteit'] = 'Opladen einde rit'
        lastrow['Datum'] = lastrow['Begindatum en -tijd'].date()
        lastrow['verbruik'] = 0
        lastrow['energie'] = eindstand
        lastrow['Laden'] = 1
        lastrow['nacht'] = 0
        lastrow['bijladen'] = (battery - eindstand)
        lastrow['Duur'] = aansluittijd + lastrow['bijladen']/laadvermogen*3600
        lastrow['Einddatum en -tijd'] = lastrow['Begindatum en -tijd'] + timedelta(seconds = lastrow['Duur'])
           
        try:
            gemiddelde_prijs = sum(df['Laadkosten (EUR)'])/sum(df['bijladen'])*1000
        except:
            prices_filter = prices.loc[prices['datetime_CET_end'] > lastrow['Begindatum en -tijd']]
            prices_filter = prices_filter.loc[prices_filter['datetime_CET'] < lastrow['Einddatum en -tijd']]
            gemiddelde_prijs = np.mean(prices_filter['price_eur_mwh'])
        lastrow['price_eur_mwh'] = gemiddelde_prijs
        # TODO: laadkosten laatste regel worden nu berekend op uurprijs van de start van de activiteit
        lastrow['Laadkosten (EUR)'] = lastrow['bijladen']*lastrow['price_eur_mwh']/1000
        lastrow['Laadkosten_snel (EUR)'] = 0
        lastrow['Gemiddelde laadprijs (EUR/kWh)'] = lastrow['price_eur_mwh']/1000
        # Append the modified row
        df_result = pd.concat([df_result, pd.DataFrame([lastrow])], ignore_index=True)
    
    return df_result

@st.cache_data
# Simulatie voor het laadmodel MET de optie voor bijladen langs de snelweg
# TODO: bepalen of deze functie overbodig is
def simulate2(df2, zuinig = [1.26], laadvermogen = [44], aansluittijd = [600], battery = [540], nachtladen = 0, activiteitenladen = 0, type_voertuigen = 1):

    if (nachtladen == 0) & (activiteitenladen == 0):
    	df2['Laadtijd'] = np.where((df2['thuis'] == 1), df2['Duur'],0) # alleen thuis laden
    elif (nachtladen == 1) & (activiteitenladen == 0):
    	df2['Laadtijd'] = np.where((df2['thuis'] == 1) | (df2['nacht'] == 1), df2['Duur'],0) # thuis of 's nachts laden
    elif (nachtladen == 0) & (activiteitenladen == 1):
    	df2['Laadtijd'] = np.where((df2['thuis'] == 1) | (df2['Laden'] == 1), df2['Duur'],0) # thuis of tijdens activiteit
    elif (nachtladen == 1) & (activiteitenladen == 1):
    	df2['Laadtijd'] = np.where((df2['thuis'] == 1) | (df2['Laden'] == 1) | (df2['nacht'] == 1), df2['Duur'],0) # thuis, 's nachts of tijdens activiteit 		
		
    type_voertuig = np.minimum(int(df2['Type voertuig'].min()-1), type_voertuigen - 1)
        
    battery = battery[type_voertuig]
    zuinig = zuinig[type_voertuig]
    laadvermogen = laadvermogen[type_voertuig]
    aansluittijd = aansluittijd[type_voertuig]
        
    df2['Laadtijd'] = np.where((df2['Activiteit'] == 'Rijden') | (df2['Afstand'] >= 3), 0, df2['Laadtijd']) # niet AC-laden tijdens rijden		
    df2['laad_potentiaal1'] = np.minimum(battery, np.maximum(0,(df2['Laadtijd'] - df2['Aansluittijd'])/3600)*laadvermogen)
    #df2['laad_potentiaal1'] =df2.apply(lambda x: min(battery,
    #                                max(0,((x['Laadtijd']-x['Aansluittijd'])/3600))*laadvermogen))
    df2 = df2.merge(tekort_snel(df2, battery = battery, zuinig = zuinig), left_index = True, right_index = True, how = 'left')
    df2['bijladen_snel'] = df2['bijladen_snel'].fillna(0)
    df2['bijladen'] = df2['bijladen'].fillna(0)
   
    energy = [battery]
    verbruik = []
    bijladen = []
    bijladen_snel = []
    laad_potentiaal = []
    laadtijd = []
    aansluittijd = []

    for i in range(df2.shape[0]):
        laad_potentiaal_update = df2.iloc[i]['laad_potentiaal1']
        laad_potentiaal.append(laad_potentiaal_update)
        laadtijd_update = df2.iloc[i]['Laadtijd']
        laadtijd.append(laadtijd_update)
        aansluittijd_update = df2.iloc[i]['Aansluittijd']
        aansluittijd.append(aansluittijd_update)
        
        verbruik_update = -df2.iloc[i]['Afstand']*zuinig
        energie_update = energy[i] + verbruik_update
        
        bijladen_snel_update = df2.iloc[i]['bijladen_snel']
        energie_update = energie_update + bijladen_snel_update
        bijladen_snel.append(bijladen_snel_update)
        
        bijladen_update = df2.iloc[i]['bijladen']
        energie_update = energie_update + bijladen_update
        bijladen.append(bijladen_update)
        
        verbruik.append(verbruik_update)
        energy.append(energie_update)
    
    return_df = pd.DataFrame({'energie' : energy[:-1],
                              'laad_potentiaal': laad_potentiaal,
                              'aansluittijd': aansluittijd,
                              'laadtijd': laadtijd,
                              'verbruik': verbruik,
                             'bijladen' : bijladen,
                             'bijladen_snel' : bijladen_snel,
    						 'index' : df2['index']}, index = df2.index)
    return return_df

@st.cache_data
# Simulatie voor het laadmodel ZONDER de optie voor bijladen langs de snelweg
def simulate(df2, zuinig = [1.26], laadvermogen = [44], aansluittijd = [600], battery = [540], nachtladen = 0, activiteitenladen = 0, type_voertuigen = 1, snelwegladen = 0):    

    if (nachtladen == 0) & (activiteitenladen == 0):
    	df2['Laadtijd'] = np.where((df2['thuis'] == 1), df2['Duur'],0) # alleen thuis laden
    elif (nachtladen == 1) & (activiteitenladen == 0):
    	df2['Laadtijd'] = np.where((df2['thuis'] == 1) | (df2['nacht'] == 1), df2['Duur'],0) # thuis of 's nachts laden
    elif (nachtladen == 0) & (activiteitenladen == 1):
    	df2['Laadtijd'] = np.where((df2['thuis'] == 1) | (df2['Laden'] == 1), df2['Duur'],0) # thuis of tijdens activiteit
    elif (nachtladen == 1) & (activiteitenladen == 1):
    	df2['Laadtijd'] = np.where((df2['thuis'] == 1) | (df2['Laden'] == 1) | (df2['nacht'] == 1), df2['Duur'],0) # thuis, 's nachts of tijdens activiteit

    df2['Laadtijd'] = np.where((df2['Activiteit'] == 'Rijden') | (df2['Afstand'] >= 3), 0, df2['Laadtijd']) # niet AC-laden tijdens rijden  		

    type_voertuig = np.minimum(int(df2['Type voertuig'].min()-1), type_voertuigen - 1)

    battery = battery[type_voertuig]
    zuinig = zuinig[type_voertuig]
    laadvermogen = laadvermogen[type_voertuig]
    aansluittijd = aansluittijd[type_voertuig]
        
    energy = [battery]
    verbruik = []
    bijladen = []
    bijladen_snel = []
    
    for i in range(df2.shape[0]):
    
        verbruik_update = -df2.iloc[i]['Afstand']*zuinig
        energie_update = energy[i] + verbruik_update
        
        bijladen_update = min(laadvermogen*(max(0, df2.iloc[i]['Laadtijd']-df2.iloc[i]['Aansluittijd'])/3600), battery - energie_update)
        energie_update = energie_update + bijladen_update
        
        bijladen_snel_update = max(0, -energie_update*snelwegladen)
        energie_update = energie_update + bijladen_snel_update
        bijladen_snel.append(bijladen_snel_update)
        
        bijladen.append(bijladen_update)
        verbruik.append(verbruik_update)
        energy.append(energie_update)
        
    return_df = pd.DataFrame({'energie' : energy[:-1],
                              'verbruik': verbruik,
                             'bijladen' : bijladen,
                             'bijladen_snel' : bijladen_snel,
							 'index' : df2['index']}, index = df2.index)

    return return_df
	
@st.cache_data
def charge_quarter(df, laadvermogen = [44], aansluittijd = [600], battery = [540], smart = 0):
    df_bijladen = df.loc[df.bijladen > 0].copy()
    show_figure = True
    
    if df_bijladen.empty:
        show_figure = False
    else:
        # Corrigeer de begindatum en -tijd voor aansluittijd
        df_bijladen['Aansluittijd_totaal'] = df_bijladen.apply(lambda g: pd.to_timedelta(aansluittijd[g['Type voertuig']-1], unit = 's'), axis=1)
        df_bijladen['Begindatum en -tijd'] = df_bijladen['Begindatum en -tijd'] + df_bijladen['Aansluittijd_totaal']
        
        # Haal het laadvermogen van het bijbehorende type voertuig op
        df_bijladen['Laadvermogen'] = df_bijladen.apply(lambda g: laadvermogen[g['Type voertuig']-1], axis=1)
    
        # Splits de activiteit op in blokken van 15 min.
        df_bijladen['StartTimeRound'] = df_bijladen['Begindatum en -tijd'].dt.floor('15min')
        df_bijladen['EndTimeRound'] = df_bijladen['Einddatum en -tijd'].dt.floor('15min')
        df_bijladen['Quarter'] = df_bijladen.apply(lambda row: list(pd.date_range(start = row['StartTimeRound'], 
                                                            end = row['EndTimeRound'], 
                                                            freq = '15min')), axis = 1)
        
        df_bijladen = df_bijladen.explode(column = 'Quarter').reset_index()[['index','Voertuig','Type voertuig', 'Quarter', 'Begindatum en -tijd', 'Einddatum en -tijd', 'bijladen','thuis', 'Laadvermogen']]
    
        # Bereken start- en eindtijd voor ieder blok (eerste en laatste blok worden gecorrigeerd voor start en eind activiteit)
        df_bijladen['StartTime'] = pd.to_datetime(df_bijladen['Quarter'])
        df_bijladen['EndTime'] = df_bijladen['StartTime'] + pd.Timedelta(minutes = 15)
        df_bijladen['StartTime'] = df_bijladen[['StartTime', 'Begindatum en -tijd']].max(axis=1)
        df_bijladen['EndTime'] = df_bijladen[['EndTime', 'Einddatum en -tijd']].min(axis=1)
        
        # Bereken hoeveel seconden er geladen kan worden
        df_bijladen['TotalSeconds'] = (df_bijladen['Einddatum en -tijd'] - df_bijladen['Begindatum en -tijd']).dt.total_seconds()
        df_bijladen['SecondsQuarter'] = (df_bijladen['EndTime'] - df_bijladen['StartTime']).dt.total_seconds()
        df_bijladen['SecondsBeforeQuarter'] = (df_bijladen['StartTime'] - df_bijladen['Begindatum en -tijd']).dt.total_seconds()
        
        if smart == 0:
            # Laad op maximaal vermogen totdat de hoeveelheid kWh bijladen is bereikt
            df_bijladen['bijladen'] = np.maximum(0,np.minimum(df_bijladen['Laadvermogen']*df_bijladen['SecondsQuarter']/3600, df_bijladen['bijladen'] - df_bijladen['Laadvermogen']*df_bijladen['SecondsBeforeQuarter']/3600))
        elif smart == 1:
            # Smeer de hoeveelheid kWh bijladen gelijkmatig uit over de activiteit
            df_bijladen['bijladen'] = df_bijladen['bijladen']*df_bijladen['SecondsQuarter']/df_bijladen['TotalSeconds']
        df_bijladen['Hour'] = df_bijladen['StartTime'].dt.hour
        df_bijladen['Date'] = df_bijladen['StartTime'].dt.date
        df_bijladen['Time'] = df_bijladen['Quarter'].dt.time

    return df_bijladen, show_figure

def check_file(file):
	
    xl = pd.ExcelFile(file)
    sheetnames = xl.sheet_names  # see all sheet names

    # Check if the DataFrame has the required column name
    if sorted(sheetnames) != ['laadlocaties', 'laden', 'ritten']:
        error_message = 'het inputbestand moet sheets bevatten met de namen "ritten", "laden" en "laadlocaties". Gebruik het template als voorbeeld.'
        st.error(error_message)
        st.stop()
		
    elif list(pd.read_excel(file, sheet_name = 'laden').columns) != ['Activiteit']:
        error_message = 'De sheet "laden" moet alleen een kolom "Activiteit" bevatten met de activiteiten waarvoor laden is toegestaan (zie nieuw template).'
        st.error(error_message)
        st.stop()

@st.cache_data
def process_excel_file(file, battery, zuinig, aansluittijd, laadvermogen, laadvermogen_snel, nachtladen, activiteitenladen, snelwegladen, laadprijs_snelweg, type_voertuigen = 1):

    # Read the Excel file into a DataFrame
    df = pd.read_excel(file, sheet_name = 'ritten')
    df['Positie'] = df['Positie'].fillna('onbekend')
    df['Afstand'] = df['Afstand'].fillna(0)
	
    optional_columns = df.columns.difference(['Voertuig', 'Type voertuig', 'Begindatum en -tijd','Einddatum en -tijd','Positie', 'Afstand', 'Activiteit'])
	
    # data cleaning
    df["Begindatum en -tijd"] = pd.to_datetime(df['Begindatum en -tijd'])
    df["Einddatum en -tijd"] = pd.to_datetime(df['Einddatum en -tijd'])
    
    df['Type voertuig'] = np.minimum(df['Type voertuig'], type_voertuigen).astype(int)

    df = df.sort_values(['Voertuig', 'Begindatum en -tijd']).reset_index(drop = True)

    # fill gaps in the time series with 'Rusten' activity
    df['lag'] = df.groupby('Voertuig')['Einddatum en -tijd'].shift(1)
    mask = (df['Begindatum en -tijd'] !=  df['lag']) & df['lag'].notna()
   
    rows = df.loc[mask].drop(columns=['Einddatum en -tijd'])
    rows = rows.rename(columns={'Begindatum en -tijd': 'Einddatum en -tijd', 'lag': 'Begindatum en -tijd'})
    rows['Activiteit'] = 'Rusten'
    rows['Afstand'] = 0
        
    # Append the new rows and sort the index
    df = pd.concat([df, rows], axis = 0).drop(columns=['lag'])
    
    df = df.sort_values(['Voertuig', 'Begindatum en -tijd']).reset_index(drop = True)

    #set which activities can be used for charging
    df_act = pd.read_excel(file, sheet_name = 'laden').assign(Laden = 1)
    df = df.merge(df_act, how = 'left', on = 'Activiteit')
    df['Laden'] = df.Laden.fillna(0)
	
    if df.Voertuig.nunique() == 1: 
        df['activiteit_g'] = (df['Activiteit'] != df.shift().fillna(method='bfill')['Activiteit']).cumsum()
    else: 
        df['activiteit_g'] = (df.groupby('Voertuig',group_keys=False).
                      apply(lambda g: (g['Activiteit'] != g.shift().fillna(method='bfill')['Activiteit']).cumsum()))
    agg_dict = {'Type voertuig': 'first',
                'Activiteit' : 'first',
                'Positie' : 'first',
                'Afstand' : 'sum',
                'Begindatum en -tijd' : 'min',
                'Einddatum en -tijd' : 'max',
                'Laden' : 'first'}
    
    optional_dict = {key : 'first' for key in optional_columns}
    agg_dict = {**agg_dict, **optional_dict}
	
    df = df.groupby(['Voertuig','activiteit_g']).agg(agg_dict).reset_index(drop = False).drop('activiteit_g', axis = 1)
    df['Activiteit_id'] = df.index
    
    df['Duur_nacht'] = (df['Einddatum en -tijd'] - df['Begindatum en -tijd']).apply(lambda x: x.total_seconds())
    df['nacht'] = np.where(((df.Afstand < 3) & (df.Duur_nacht > 6*3600)),1,0)
    
    # Prijzen inladen
    # Laad prijzen in vanuit Excelbestand in repository
    prices_path = 'day_ahead_prices.xlsx'
    prices = pd.read_excel(prices_path, engine='openpyxl')
        
    # Haal prijzen op via ENTSO-E API wanneer geen prijzen beschikbaar zijn
    # TODO: variabele instelbaar maken via app. Vaste prijs of variabele prijs
    variable_price = True
    datums_uniek = unique_dates(df)

    if variable_price:
        for i in datums_uniek:
            i = pd.Timestamp(i).tz_localize(None)
            if not (prices['datetime_CET'].dt.date == i.date()).any():
                print(f"Prijzen voor {i} ontbreken — ophalen via ENTSO-E API")
                # Probeer prijzen voor datum i op te halen op ENTSO-E API
                try:
                    a = get_day_ahead_prices(i.strftime('%Y%m%d'))
                    a['datetime_CET'] = a['datetime_CET'].dt.tz_localize(None)
                    a['datetime_UTC'] = a['datetime_UTC'].dt.tz_localize(None)
                    print(f'Prices found for {i}')
                    
                    # Voeg prijsdata samen met bestaande prijsdata
                    prices = pd.concat([prices,a]).sort_values(by = 'datetime_CET')
                # Wanneer API fout geeft, throw exception
                except:
                    print(f'No day ahead prices available for {i}')
        github_token = st.secrets["api_keys"]["gh_token"]
        upload_price_data(github_token, prices)
    else:
        # Werk met een vaste elektriciteitsprijs, gegeven door het bedrijf of een standaardwaarde
        print('fixed_prices')


    # Maak relevante kolommen aan voor prices tabel
    prices = prices[['datetime_CET', 'price_eur_mwh']].sort_values(by = 'datetime_CET')
    prices['datetime_CET_end'] = prices['datetime_CET'].shift(-1)
    prices['Datum'] = prices['datetime_CET'].dt.date
    # Forward fill prijzen die leeg zijn
    prices['price_eur_mwh'] = prices['price_eur_mwh'].ffill()
    
    # Prijzen mergen aan het dataframe
    df = add_day_ahead_prices(df, prices, aansluittijd)
    df['Duur'] = (df['Einddatum en -tijd'] - df['Begindatum en -tijd']).apply(lambda x: x.total_seconds())
    df_locatie = pd.read_excel(file, sheet_name = 'laadlocaties').assign(thuis = 1)
    df = df.merge(df_locatie, how = 'left', on = 'Positie')

    df['thuis'] = df.thuis.fillna(0)
    df = df.sort_values(['Voertuig', 'Begindatum en -tijd']).reset_index()

    df_results = (df.
    			groupby(['Voertuig']).
    			apply(lambda g: simulate(g, battery = battery, zuinig = zuinig, aansluittijd = aansluittijd, laadvermogen = laadvermogen, nachtladen = nachtladen, activiteitenladen = activiteitenladen, type_voertuigen = type_voertuigen, snelwegladen = snelwegladen)))

    df = df.merge(df_results, on = 'index', how = 'left')
    
    # Groepeer op activiteit_id om gesplitste rijen voor uur/kwartierprijzen terug te brengen naar 1 activiteit
    df = aggregate_hourly_costs(df)
    
    df['Laadkosten_snel (EUR)'] = df['bijladen_snel']*laadprijs_snelweg

    #TODO: fix aansluittijd naar de aansluittijd die bij het type voertuig hoort
    df['vertraging'] = np.where(df['bijladen_snel'] > 0, aansluittijd[0] + (3600*df['bijladen_snel']/laadvermogen_snel),0)

    # Voeg een extra regel toe voor ieder voertuig wanneer extra bijladen nodig is
    df = df.groupby('Voertuig').apply(lambda g: bijladen_einde_rit(g, prices, laadvermogen = laadvermogen, battery = battery, aansluittijd = aansluittijd, type_voertuigen = type_voertuigen), include_groups = False)
    #df = df.reset_index(level=0, drop=True).reset_index()
    df = df.reset_index(level=1, drop=True).reset_index()
    df = df.drop('index', axis = 1)
    
    df = df[['Voertuig','Type voertuig', 'Activiteit', 'Begindatum en -tijd', 'Einddatum en -tijd', 'Positie', 'Afstand', 'Laden', 'Duur', 'nacht', 'thuis', 'energie', 'verbruik', 'bijladen', 'bijladen_snel', 'Laadkosten (EUR)', 'Laadkosten_snel (EUR)', 'Gemiddelde laadprijs (EUR/kWh)', 'vertraging']]
    return df


def show_haalbaarheid(df):
    #df['Datum'] = df.groupby(['Voertuig','RitID'])['Begindatum en -tijd'].transform(lambda x: x.min().date())
    df['Datum'] = df['Begindatum en -tijd'].dt.date

    haalbaarheid = df.pivot_table(values = 'energie', index = 'Voertuig', columns = ['Datum'], aggfunc=lambda x: 1 if min(x) >= -0.01 else 0)
    cmap=LinearSegmentedColormap.from_list('rg',["r","y", "g"], N=256) 
    st.subheader('Haalbaarheid van de planning per dag en voertuig')
    fig1 = plt.figure(figsize=(10, 4))
    sns.heatmap(haalbaarheid, annot=False, cmap = cmap, vmin=0, vmax=1, linewidths=1, linecolor='black', cbar = False)
	
    st.pyplot(fig1)
 
def show_demand_table(df):
    st.subheader('Geladen energie per locatie (top 10)')
    bijladen = df.groupby('Positie').bijladen.sum().reset_index()
    bijladen = pd.concat([bijladen,pd.DataFrame({'Positie' : ['Snelweg'],
	    'bijladen' : [df.bijladen_snel.sum()]})]).sort_values(by = 'bijladen', ascending = False).rename(columns = {'bijladen': 'Hoeveelheid energie geladen (kWh)'})
    st.table(bijladen.reset_index(drop = True).loc[lambda d: d['Hoeveelheid energie geladen (kWh)'] >0].head(10))


def plot_demand(df, battery, zuinig, aansluittijd, laadvermogen, laadvermogen_snel, type_voertuigen = 1):
    
    st.subheader('De gemiddelde verdeling van de energievraag over de dag')
    
    charge_locations = df.loc[lambda d: d.bijladen >0].groupby('Positie').bijladen.sum().sort_values(ascending = False).index
    charge_locations = pd.Index(['Totaal']).append(charge_locations)
    filter_option = st.selectbox('Selecteer een locatie', charge_locations, index = 0)
    
    @st.cache_data
    def filter_data(df, selected_option, smart = 0, perc = 95):
        df_plot = df.copy()
        df_plot['Date'] = df_plot['Begindatum en -tijd'].dt.date
        df_plot['Date_end'] = df_plot['Einddatum en -tijd'].dt.date
        
        date_min = df_plot.loc[df_plot['Activiteit']!='Opladen einde rit', 'Date'].min()
        date_max = df_plot.loc[df_plot['Activiteit']!='Opladen einde rit', 'Date_end'].max()

        if selected_option == 'Totaal':
            df_plot = df.copy()    
        else:
            df_plot = df.loc[df.Positie == selected_option]
        
        
        plot_data1, show_figure = charge_quarter(df_plot, smart = smart, battery = battery, aansluittijd = aansluittijd, laadvermogen = laadvermogen)
        
        if show_figure:
            plot_data1 = (plot_data1.groupby(['Date', 'Time']).bijladen.sum()).reset_index()
            uniques = [pd.date_range(start='00:00', end='23:45', freq='15min').time, pd.date_range(date_min, date_max, freq = '1d')]

            df_hour_date = pd.DataFrame(product(*uniques), columns = ['Time', 'Date'])
            df_hour_date['Date'] = df_hour_date['Date'].dt.date

            plot_data1 = df_hour_date.merge(plot_data1, how = 'left', on = ['Date', 'Time']).fillna(0)
            
            plot_data1['Time_num'] = (
                plot_data1['Time'].apply(lambda t: t.hour + t.minute/60 + t.second/3600)
            )
            plot_data1['vermogen'] = plot_data1['bijladen']*4
            plot_data1 = plot_data1.groupby('Time')['vermogen'].agg(mean='mean', median='median', p=lambda x: x.quantile(perc/100)).reset_index()
        else:
            plot_data1 = pd.DataFrame()
        return plot_data1, show_figure
		
    percentile_choice = st.radio('Percentiel waarde voor error bar',[75,85,95], index=2)

	# Create a demand plot
    #fig, (ax1) = plt.subplots(1, 1, figsize=(12, 6))
    fig = go.Figure()

    plot_data, show_figure = filter_data(df, filter_option, 0, percentile_choice)

    if show_figure:
        fig.add_trace(go.Scatter(x=plot_data['Time'], y=plot_data['mean'], mode = 'lines', name = 'Normaal laden: Gemiddelde', line_color = '#4248f5'))
        fig.add_trace(go.Scatter(x=plot_data['Time'], y=plot_data['p'], mode = 'lines', fill = 'tozeroy', name = f'Normaal laden: {percentile_choice}%-percentiel', line_color = "rgba(141, 164, 247, 0)", fillcolor = "rgba(141, 164, 247, 0.2)"))

        plot_data_smart, _ = filter_data(df, filter_option, 1, percentile_choice)
        fig.add_trace(go.Scatter(x=plot_data_smart['Time'], y=plot_data_smart['mean'], mode = 'lines', name = 'Slim laden: Gemiddelde', line_color = '#c41f30'))
        fig.add_trace(go.Scatter(x=plot_data_smart['Time'], y=plot_data_smart['p'], mode = 'lines', fill = 'tozeroy', name = f'Slim laden: {percentile_choice}%-percentiel', line_color = "rgba(252, 149, 158, 0)", fillcolor = "rgba(252, 149, 158, 0.2)"))

        fig.update_layout(
            title="Vermogensvraag per uur",
            title_x = 0.5,
            title_xanchor = 'center',
            title_font = dict(size=24),
            xaxis_title="Tijd (Uren)",
            yaxis_title="Gevraagd vermogen (kW)",
            xaxis = dict(
                tickmode = 'linear',
                tick0 = 0,
                dtick = 12,
                tickformat = "%H:%M"
                ),
            legend=dict(
                orientation="h",    
                yanchor="top",
                y=-0.3, 
                xanchor="center",
                x=0.5,
                traceorder="normal",
                )
            )

        st.plotly_chart(fig, use_container_width=True)
    

        # Offer the file download
        
            
        # Create a BytesIO object
        excel_data = BytesIO()

        # Save the DataFrame to BytesIO as an Excel file
        with pd.ExcelWriter(excel_data, engine='xlsxwriter') as writer:
            for location in charge_locations:              
                temp, _ = filter_data(df, location, 0, percentile_choice)
                temp.rename({'p': f'percentile_{percentile_choice}'})
                temp.to_excel(writer, index=True, sheet_name = location[:15]+"_normaal_laden")
                temp1, _ = filter_data(df, location, 1, percentile_choice)
                temp1.rename({'p': f'percentile_{percentile_choice}'})
                temp1.to_excel(writer, index=True, sheet_name = location[:15]+"_slim_laden")
        # Set the BytesIO object's position to the start
        excel_data.seek(0)
            
        st.download_button('Download data', excel_data, file_name='data_demand_plot.xlsx')
    
    else:
        st.write('Er zijn geen mogelijkheden voor uw voertuigen om te laden. Daarom kan de vermogensvraag niet worden gevisualiseerd.')


def download_excel(df):

    st.subheader('Definitieve dataset')
	
    @st.cache_data
    def create_output_excel(df):
        df = df.drop(['Datum'], axis = 1)
        excel_data = BytesIO()
        with pd.ExcelWriter(excel_data, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Sheet1')
        excel_data.seek(0)
        return excel_data

    # Offer the file download
    st.download_button('Download Excelbestand met modeluitkomsten', create_output_excel(df), file_name='laadmodel_resultaten.xlsx')

def download_template():
    
 
    with open('template.xlsx', "rb") as f:
        bytes_data = f.read()
 
    st.download_button('Download Template', bytes_data, file_name='template.xlsx')

def table_kosten(df, energiebelasting = 0.00321, laadprijs_snelweg = 0.74, type_voertuigen = 1, verbruik_diesel = [25, 18, 8]):
    
    # Let op: energiebelasting is afhankelijk van jaarverbruik: https://www.belastingdienst.nl/wps/wcm/connect/bldcontentnl/belastingdienst/zakelijk/overige_belastingen/belastingen_op_milieugrondslag/energiebelasting/
    laadkosten_epex = sum(df['Laadkosten (EUR)'])
    laadkosten_snelweg = sum(df['Laadkosten_snel (EUR)'])
    #energiebelasting_totaal = sum(df['bijladen'])*energiebelasting + sum(df['bijladen_snel'])*energiebelasting
    
    bijladen = sum(df['bijladen'])
    bijladen_snelweg = sum(df['bijladen_snel'])
    
    tabel = pd.DataFrame({
        'Categorie': ['Laadkosten EPEX', 'Laadkosten snelweg'], #'Energiebelasting'
        'Gevraagde energie (kWh)' : [f"{format_nl_smart(bijladen,0)}",f"{format_nl_smart(bijladen_snelweg,0)}"],
        'Kosten': [f"€{format_nl_smart(laadkosten_epex,0)}", f"€{format_nl_smart(laadkosten_snelweg,0)}"], #f"€{format_nl_smart(energiebelasting_totaal,0)
        'Aanname': ['Op basis van variabele day-aheadprijzen', f'Vaste prijs voor laden snelweg van €{format_nl_smart(laadprijs_snelweg)}/kWh'] #, f'Uitgaande van het tarief van €{format_nl_smart(energiebelasting,5)}/kWh'
        })
    
    # Haal de dieselprijzen op uit het Excelbestand. Let op! Deze dient ververst te worden voor huidige dieselprijzen via https://www.tln.nl/ledenvoordeel/brandstofmonitor
    diesel_path = 'Dagelijkse-dieselprijs.xlsx'
    prices_diesel = pd.read_excel(diesel_path, engine='openpyxl')
    prices_diesel['Datum'] = pd.to_datetime(prices_diesel["Datum"])
    df['Datum'] = pd.to_datetime(df['Datum'])
    
    # Neem het gemiddelde van de laatste 10 weken aan als er geen dieselprijs gekoppeld kan worden
    price_diesel_avg = prices_diesel["Dieselprijs"].tail(70).mean().item()
    
    df = pd.merge(df, prices_diesel, on = 'Datum', how = 'left')
    
    # Vul alle lege waarden met de gemiddelde waarde van dat jaar
    mask = df["Dieselprijs"].isna()
    df.loc[mask, "Dieselprijs"] = price_diesel_avg
    
    # Vermenigvuldig dieselprijs met verbruik van een diesel.
    # Aannames:
    # Vrachtwagen: 25L/100km
    # Bakwagen: 18L/100km (op basis van de ratio e-verbruik bakwagen/e-verbruik vrachtwagen)
    # Bestelwagen: 8L/100km (op basis van de ratio e-verbruik bestelwagen/e-verbruik vrachtwagen)
    df['Kosten diesel']  = df.apply(lambda row: row['Dieselprijs']*row['Afstand']*verbruik_diesel[row['Type voertuig']-1]/100, axis=1)

    voertuigen = pd.DataFrame({
        'Type voertuig': [1,2,3],
        'Voertuig': ['Trekker-oplegger (N3)', 'Bakwagen (N2)', 'Bestelwagen (N1)'],
        'Aanname verbruik': verbruik_diesel
    }).set_index('Type voertuig')

    tab_comp = df.groupby('Type voertuig').agg({'Afstand': 'sum',
                                                         'bijladen': 'sum', 
                                                         'bijladen_snel': 'sum', 
                                                         'Laadkosten (EUR)': 'sum', 
                                                         'Laadkosten_snel (EUR)': 'sum', 
                                                         'Kosten diesel': 'sum'})
    
    tab_comp['Laadvraag (kWh)'] = tab_comp['bijladen'] + tab_comp['bijladen_snel']
    tab_comp['Kosten elektrisch'] = tab_comp['Laadkosten (EUR)'] + tab_comp['Laadkosten_snel (EUR)']

    #Let op delen door 0 fouten
    tab_comp['Laadkosten per km'] = tab_comp['Kosten elektrisch']/tab_comp['Afstand']
    tab_comp['Dieselkosten per km'] = tab_comp['Kosten diesel']/tab_comp['Afstand']

    tab_comp = tab_comp.join(voertuigen, how = 'left').reset_index()

    tab_comp = tab_comp[['Voertuig', 'Afstand', 'Laadvraag (kWh)', 'Kosten elektrisch', 'Kosten diesel']]
    tab_comp = tab_comp.rename({'Afstand': 'Afstand (km)'}, axis = 1)

    kosten_diesel = sum(df['Kosten diesel']) 
    laadkosten_totaal = laadkosten_epex + laadkosten_snelweg # + energiebelasting_totaal
    totale_kms = sum(df['Afstand'])

    try:
        kosten_per_km = laadkosten_totaal/totale_kms
        kosten_diesel_per_km = kosten_diesel/totale_kms
    except ZeroDivisionError:
        kosten_per_km = 0
        kosten_diesel_per_km = 0

    return tabel, laadkosten_totaal, totale_kms, kosten_per_km, kosten_diesel, kosten_diesel_per_km, tab_comp

def format_nl_smart(x, decimals = 2):
    import locale
    locale.setlocale(locale.LC_ALL, 'C')
    if pd.isna(x):
        return ""
    if float(x).is_integer():
        # Geheel getal → geen decimalen
        x = float(x)
        return f'{int(x):,}'.replace(",", ".")
    else:
        fmt = f"{{x:,.{decimals}f}}"
        s = fmt.format(x=x)
        return s.replace(",", "X").replace(".", ",").replace("X", ".")

def upload_price_data(API_TOKEN, df):
    
    OWNER = "DatalabHvA"
    REPO = "laadmodel"
    FILE_PATH = "day_ahead_prices.xlsx"
    
    buffer = BytesIO()
    df.to_excel(buffer, index=False, engine='openpyxl')
    xlsx_bytes = buffer.getvalue()
    content_b64 = base64.b64encode(xlsx_bytes).decode("utf-8")


    headers = {
            "Authorization": f"Bearer {API_TOKEN}",
            "Accept": "application/vnd.github+json",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache"
            }

    url = f"https://api.github.com/repos/{OWNER}/{REPO}/contents/{FILE_PATH}"

    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        sha = r.json()["sha"]
    else:
        sha = None  # nieuw bestand
    
    payload = {
        "message": "Update XLSX via API " + datetime.now().strftime("%d-%m-%Y, %H:%M:%S"),
        "content": content_b64,
    }
    
    if sha:
        payload["sha"] = sha
    
    response = requests.put(url, headers=headers, json=payload)
    print(response.status_code)
    return response.status_code

def main():
    
    cols = st.columns([0.5, 0.3, 0.2], vertical_alignment = "bottom")
    with cols[0]:
        st.title('Laadmodel ZEC')
    with cols[1]:
        with open("Handleiding_laadmodel_ZEC_v2_0_0.pdf", "rb") as f:
            pdf_bytes = f.read()
    
            # Create the download button
        st.download_button(
            label="Download Handleiding",
            data=pdf_bytes,
            file_name="Handleiding_laadmodel_ZEC_v2_0_0.pdf",
            mime="application/pdf"
        )
    with cols[2]:
        st.write('v2.0.0, 30-1-2026')
    
    st.write("De resultaten van deze tool zijn informatief.  \nDe verstrekte informatie kan onvolledig of niet geheel juist zijn.  \nAan de resultaten van deze tool kunnen geen rechten worden ontleend.")

    # Parameter setting
    st.header('Instellingen laadmodel')
    st.subheader('Voertuigaannames')
    st.write('Hoeveel verschillende typen voertuigen heeft u? U kunt uw voertuigen opdelen in trekker-opleggers, bakwagens en/of bestelwagens')
    
    type_voertuigen = 1
    type_voertuigen = st.slider("Aantal typen voertuigen", min_value = 1, max_value = 3, help = 'Wanneer u kiest voor 1 type voertuigen, hebben alle voertuigen binnen het wagenpark dezelfde accucapaciteit en hetzelfde verbruik. U kunt meerdere typen aanmaken door deze slider naar 2 of 3 te verschuiven. Vervolgens geeft u in het Exceltemplate aan of een voertuig tot type 1, 2 of 3 behoort.')
    
    st.write('Gebruik de onderstaande sliders om aannames in te vullen waar het laadmodel mee rekent. De standaardwaarden zijn in samenwerking met TNO bepaald in mei 2025.')
    
    # Maak kolommen aan met gelijke breedte voor de sliders
    lst = [1] + [3/type_voertuigen] * type_voertuigen
    col = st.columns(lst)
    col1 = st.columns(lst)
    col2 = st.columns(lst)
    col3 = st.columns(lst)

    col[0].write(' ')
    col[0].write('**Capaciteit accu (kWh)**')
    col1[0].write(' ')
    col1[0].write('**Verbruik (kWh/km)**')
    col2[0].write(' ')
    col2[0].write('**Laadvermogen (kW)**')
    col3[0].write(' ')
    col3[0].write('**Aansluittijd (min)**')
    
    battery = []
    zuinig = []
    laadvermogen = []
    aansluittijd = []
        
    if type_voertuigen >= 1:
        battery_N3 = col[1].slider("Trekker-oplegger (N3)", 0, 800, 540, key = 'batteryN3', help = 'Vul hier de accucapaciteit in van het voertuig in kWh.')
        zuinig_N3 = col1[1].slider("Trekker-oplegger (N3)", 0.0, 2.5, 1.26, key = 'zuinigN3', help = 'Vul hier het verbruik van het voertuig in. Dit getal geeft aan hoeveel kWh aan elektriciteit er verbruikt wordt per gereden km.')
        laadvermogen_N3 = col2[1].slider("Trekker-oplegger (N3)", 0, 300, 44, key = 'laadvermogenN3', help = 'Vul hier het vermogen in waarmee dit voertuig kan laden aan een laadpaal bij de thuislocatie. Dit vermogen geeft aan hoeveel kWh de vrachtwagen in 1 uur kan laden.')
        aansluittijd_N3 = 60*col3[1].slider("Trekker-oplegger (N3)", 0, 20, 10, key = 'aansluittijdN3', help = 'Vul hier de tijd in die het kost om dit voertuig aan een laadpaal aan te sluiten. Neem hierbij ook het inparkeren in acht.')
        
        battery.append(battery_N3)
        zuinig.append(zuinig_N3)
        laadvermogen.append(laadvermogen_N3)
        aansluittijd.append(aansluittijd_N3)
        
    if type_voertuigen >= 2:
        battery_N2 = col[2].slider("Bakwagen (N2)", 0, 800, 350, key = 'batteryN2')
        zuinig_N2 = col1[2].slider("Bakwagen (N2)", 0.0, 2.5, 0.9, key = 'zuinigN2')
        laadvermogen_N2 = col2[2].slider("Bakwagen (N2)", 0, 300, 44, key = 'laadvermogenN2')
        aansluittijd_N2 = 60*col3[2].slider("Bakwagen (N2)", 0, 20, 10, key = 'aansluittijdN2')
        
        battery.append(battery_N2)
        zuinig.append(zuinig_N2)
        laadvermogen.append(laadvermogen_N2)
        aansluittijd.append(aansluittijd_N2)
        
    if type_voertuigen >=3 :
        battery_N1 = col[3].slider("Bestelwagen (N1)", 0, 800, 75, key = 'batteryN1')
        zuinig_N1 = col1[3].slider("Bestelwagen (N1)", 0.0, 2.5, 0.4, key = 'zuinigN1')
        laadvermogen_N1 = col2[3].slider("Bestelwagen (N1)", 0, 300, 44, key = 'laadvermogenN1')
        aansluittijd_N1 = 60*col3[3].slider("Bestelwagen (N1)", 0, 20, 10, key = 'aansluittijdN1')
        
        battery.append(battery_N1)
        zuinig.append(zuinig_N1)
        laadvermogen.append(laadvermogen_N1)
        aansluittijd.append(aansluittijd_N1)
    
    st.subheader('Prijsaannames')
    st.write('Gebruik onderstaande sliders om instellingen aan te passen voor de laadprijzen waar het model mee rekent.')
    col4 = st.columns([1, 3])
    col4[0].write(' ')
    col4[0].write('**Vaste elektriciteitsprijs laden snelweg (€/kWh)**')
    laadprijs_snelweg = col4[1].slider("", min_value = 0.0, max_value = 2.0, value = 0.74, help = 'Vul hier de laadprijs in die zou gelden als uw voertuigen niet op de eigen vestiging of bij klanten laden, maar op de snelweg. Voor deze kosten wordt een vaste prijs aangehouden door het model.')

    # Download template button
    st.header('Uploaden rittendata')
    #TODO: Volledige gebruiksvriendelijke instructie schrijven
    st.write('Gebruik het Excelformat hieronder om rittendata aan te leveren.')
    download_template()

    # File upload
    uploaded_file = st.file_uploader('Upload Excelbestand met rittendata', type=['xlsx'], help = 'De data die u aanlevert wordt niet opgeslagen door de makers van deze tool')

    # TODO: besluiten of nachtladen wordt gebruikt
    nachtladen = st.checkbox('Altijd opladen tijdens overnachting op alle locaties', help = 'Wanneer deze optie wordt geselecteerd, zal een vrachtwagen laden wanneer hij langer dan 6 uur stilstaat op een locatie.')
    #nachtladen = 0
    activiteitenladen = st.checkbox('Laden mogelijk op alle locaties', help = 'Wanneer deze optie wordt geselecteerd, zal laden niet alleen mogelijk zijn op de door u in het Excelbestand aangegeven locaties, maar op alle locaties.')
    snelwegladen = st.checkbox('Extra snelladen toestaan langs de snelweg', help = 'Wanneer deze optie wordt geselecteerd, zal het model tekorten opvangen door vrachtwagens extra aan de snelweg op te laden. Let op dat de planning hierdoor niet wordt aangepast, er wordt alleen extra laadenergie tijdens een rijactiviteit berekend.')
    

    if uploaded_file is not None:
        try:
            check_file(uploaded_file)
            laadvermogen_snel = 150
            print('Start verwerking Excelbestand')
            df = process_excel_file(uploaded_file, battery = battery, zuinig = zuinig, aansluittijd = aansluittijd, laadvermogen = laadvermogen, laadvermogen_snel = laadvermogen_snel, nachtladen = nachtladen, activiteitenladen = activiteitenladen, snelwegladen = snelwegladen, laadprijs_snelweg = laadprijs_snelweg, type_voertuigen = type_voertuigen)
            #print(df)
            print('Eind verwerking Excelbestand')
            st.header('Modelresultaten:')
            show_haalbaarheid(df)
            print('Start tonen demand table')
            show_demand_table(df)   
            print('Eind tonen demand table')
            #print(df)

            print('Start plot demand')
            plot_demand(df, battery = battery, zuinig = zuinig, aansluittijd = aansluittijd, laadvermogen = laadvermogen, laadvermogen_snel = laadvermogen_snel, type_voertuigen = type_voertuigen)
            print('Eind plot demand')

            st.subheader('Brandstofkosten: Elektrisch = €' + f"{format_nl_smart(table_kosten(df)[1],0)}" + ' | Diesel = €' + f"{format_nl_smart(table_kosten(df)[4],0)}")
            st.dataframe(table_kosten(df, laadprijs_snelweg = laadprijs_snelweg)[0], hide_index=True)
            st.write(f'De laadkosten voor het uitvoeren van {format_nl_smart(table_kosten(df)[2],0)} kilometers zijn €{format_nl_smart(table_kosten(df)[1],0)}. Dat is €{format_nl_smart(table_kosten(df)[3],3)} per km.')
            st.write(f'De dieselkosten voor het uitvoeren van deze ritten zijn €{format_nl_smart(table_kosten(df)[4],0)}. Dat is €{format_nl_smart(table_kosten(df)[5],3)} per km. Hieronder splitsen we deze kosten uit naar type voertuig. Deze tabel bevat ook onze aannames over het verbruik van dieselvoertuigen. Voor de dieselprijs gaan we uit van de Gemiddelde Landelijke Adviesprijs (GLA) en we gaan uit van een verbruik van 25L/100km.')
            st.dataframe(table_kosten(df, laadprijs_snelweg = laadprijs_snelweg)[6].style.format({
                'Afstand (km)': lambda x: format_nl_smart(x, 0),
                'Laadvraag (kWh)': lambda x: format_nl_smart(x, 0),
                'Kosten elektrisch': lambda x: '€' + format_nl_smart(x, 0),
                'Kosten diesel': lambda x: '€' + format_nl_smart(x, 0),
                'Aanname verbruik': lambda x: format_nl_smart(x,0) + 'L/100km'
                }), hide_index=True)
            st.write('Let op: dit betreft uitsluitend de kale brandstofkosten, voor een complete vergelijking tussen elektrisch vervoer en dieselvrachtwagens raden wij aan gebruik te maken van een tool die de Total Cost of Ownership (TCO) berekent.')
            st.markdown("[TCO-tool Topsector Logistiek](https://topsectorlogistiek.nl/tco-vracht/)")
            
            st.subheader('De eerste 15 regels van het outputbestand')
            st.dataframe(df.drop(columns=['Datum']).head(15).style.format({
                'Type voertuig': format_nl_smart,
                'Afstand': format_nl_smart,
                'Laden': format_nl_smart,
                'bijladen': format_nl_smart,
                'bijladen_snel': format_nl_smart,
                'Laadkosten (EUR)': format_nl_smart,
                'Laadkosten_snel (EUR)': format_nl_smart,
                'Gemiddelde laadprijs (EUR/kWh)': format_nl_smart,
                'Duur': format_nl_smart,
                'thuis': format_nl_smart,
                'energie': format_nl_smart,
                'verbruik': format_nl_smart,
                'vertraging': format_nl_smart
                }))
            
            download_excel(df)
        except Exception as e:
            st.error(f'Error processing the file: {e}')


if __name__ == '__main__':
    main()
