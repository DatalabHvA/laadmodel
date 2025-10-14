import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from  matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import requests
from io import BytesIO
from datetime import timedelta
import numpy as np
from itertools import product

@st.cache_data
def tekort_snel(df2, battery = 300, zuinig = 1.25):
    verbruik_rit = df2.Afstand.sum()*zuinig

    df2 = df2.sort_values('Begindatum en -tijd', ascending = False)
    tekort2 = []
    tekort1 = []

    for i, row in df2.reset_index().iterrows():
        tekort = min(min(df2.iloc[i].laad_potentiaal1,
                     max(0,battery + df2.iloc[0:i+1].Afstand.sum()*zuinig - sum(tekort1) - sum(tekort2))),
                     max(0,verbruik_rit - sum(tekort1) - sum(tekort2)))
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
    
    # TODO: zorg ervoor dat tijdstippen waar geen prijs voor bestaat niet uit de data worden gehaald
    return result

@st.cache_data
# Functie voor het koppelen van prijzen aan het df
def add_day_ahead_prices(df, prices, aansluittijd = 600):    
    
    df_laden = df[df['Laden']==1]
    df_laden = pd.concat((match_prices(row, prices) for _, row in df_laden.iterrows()), ignore_index=True)
    df_laden['Aansluittijd'] = df_laden.apply(lambda g: min(aansluittijd, (g['datetime_CET_end'] - g['Begindatum en -tijd']).total_seconds(), max(aansluittijd - (g['datetime_CET'] - g['Begindatum en -tijd']).total_seconds(), 0)), axis = 1)
    df_laden['Begindatum en -tijd'] = df_laden[['Begindatum en -tijd', 'datetime_CET']].max(axis = 1)
    df_laden['Einddatum en -tijd'] = df_laden[['Einddatum en -tijd', 'datetime_CET_end']].min(axis = 1)
    df_laden = df_laden[['Voertuig', 'Activiteit_id', 'Begindatum en -tijd', 'Einddatum en -tijd', 'Positie', 'Afstand', 'Activiteit', 'Datum', 'Laden', 'Aansluittijd', 'price_eur_mwh']]
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

    df_g = df.groupby('Activiteit_id').agg(agg_dict)#.reset_index(drop = False)#.drop('Activiteit_id', axis = 1)
    df = df_g.join([laadkosten, gemiddelde_prijs])
    print('check 3')
    df['Laadkosten (EUR)'] = df['Laadkosten (EUR)'].fillna(0)
    df['Gemiddelde laadprijs (EUR/kWh)'] = df['Gemiddelde laadprijs (EUR/kWh)'].fillna(0)
    print('check 4')
    return df

@st.cache_data
# Functie voor het toevoegen van een extra regel als aan het einde van de rit de accu nog niet terug is volgeladen
def bijladen_einde_rit(df, prices, laadvermogen = 44, battery = 540, aansluittijd = 600):
    df_result = df.copy()
    
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
        lastrow['price_eur_mwh'] = prices.loc[(prices['datetime_CET'] < lastrow['Begindatum en -tijd']) & (prices['datetime_CET_end'] > lastrow['Begindatum en -tijd']), 'price_eur_mwh'].iloc[0]
        
        # TO DO: laadkosten laatste regel worden nu berekend op uurprijs van de start van de activiteit
        lastrow['Laadkosten (EUR)'] = lastrow['bijladen']*lastrow['price_eur_mwh']/1000
        lastrow['Gemiddelde laadprijs (EUR/kWh)'] = lastrow['price_eur_mwh']/1000
        # Append the modified row
        df_result = pd.concat([df_result, pd.DataFrame([lastrow])], ignore_index=True)
    
    return df_result

@st.cache_data
# Simulatie voor het laadmodel MET de optie voor bijladen langs de snelweg
def simulate2(df2, zuinig = 1.25, laadvermogen = 44, laadvermogen_snel = 150, aansluittijd = 600, battery = 540, nachtladen = 0, activiteitenladen = 0):

    if (nachtladen == 0) & (activiteitenladen == 0):
    	df2['Laadtijd'] = np.where((df2['thuis'] == 1), df2['Duur'],0) # alleen thuis laden
    elif (nachtladen == 1) & (activiteitenladen == 0):
    	df2['Laadtijd'] = np.where((df2['thuis'] == 1) | (df2['nacht'] == 1), df2['Duur'],0) # thuis of 's nachts laden
    elif (nachtladen == 0) & (activiteitenladen == 1):
    	df2['Laadtijd'] = np.where((df2['thuis'] == 1) | (df2['Laden'] == 1), df2['Duur'],0) # thuis of tijdens activiteit
    elif (nachtladen == 1) & (activiteitenladen == 1):
    	df2['Laadtijd'] = np.where((df2['thuis'] == 1) | (df2['Laden'] == 1) | (df2['nacht'] == 1), df2['Duur'],0) # thuis, 's nachts of tijdens activiteit 		
		
    df2['Laadtijd'] = np.where((df2['Activiteit'] == 'Rijden') | (df2['Afstand'] >= 3), 0, df2['Laadtijd']) # niet AC-laden tijdens rijden  		
    df2['laad_potentiaal1'] =df2.apply(lambda x: min(battery,
                                    max(0,((x['Laadtijd']-x['Aansluittijd'])/3600))*laadvermogen))
    df2 = df2.merge(tekort_snel(df2, battery = battery, zuinig = zuinig), left_index = True, right_index = True, how = 'left')
    df2['bijladen_snel'] = df2['bijladen_snel'].fillna(0)
    df2['bijladen'] = df2['bijladen'].fillna(0)
   
    energy = [battery]
    verbruik = []
    bijladen = []
    bijladen_snel = []

    for i in range(df2.shape[0]):
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
                              'verbruik': verbruik,
                             'bijladen' : bijladen,
                             'bijladen_snel' : bijladen_snel,
    						 'index' : df2['index']}, index = df2.index)
    return return_df

@st.cache_data
# Simulatie voor het laadmodel ZONDER de optie voor bijladen langs de snelweg
def simulate(df2, zuinig = 1.25, laadvermogen = 44, laadvermogen_snel = 150, aansluittijd = 600, battery = 300, nachtladen = 0, activiteitenladen = 0):

    if (nachtladen == 0) & (activiteitenladen == 0):
    	df2['Laadtijd'] = np.where((df2['thuis'] == 1), df2['Duur'],0) # alleen thuis laden
    elif (nachtladen == 1) & (activiteitenladen == 0):
    	df2['Laadtijd'] = np.where((df2['thuis'] == 1) | (df2['nacht'] == 1), df2['Duur'],0) # thuis of 's nachts laden
    elif (nachtladen == 0) & (activiteitenladen == 1):
    	df2['Laadtijd'] = np.where((df2['thuis'] == 1) | (df2['Laden'] == 1), df2['Duur'],0) # thuis of tijdens activiteit
    elif (nachtladen == 1) & (activiteitenladen == 1):
    	df2['Laadtijd'] = np.where((df2['thuis'] == 1) | (df2['Laden'] == 1) | (df2['nacht'] == 1), df2['Duur'],0) # thuis, 's nachts of tijdens activiteit

    df2['Laadtijd'] = np.where((df2['Activiteit'] == 'Rijden') | (df2['Afstand'] >= 3), 0, df2['Laadtijd']) # niet AC-laden tijdens rijden  		

    energy = [battery]
    verbruik = []
    bijladen = []
    bijladen_snel = []
    
    for i in range(df2.shape[0]):
    
        verbruik_update = -df2.iloc[i]['Afstand']*zuinig
        energie_update = energy[i] + verbruik_update
        
        bijladen_snel_update = 0
        energie_update = energie_update + bijladen_snel_update
        bijladen_snel.append(bijladen_snel_update)
        
        bijladen_update = min(laadvermogen*(max(0, df2.iloc[i]['Laadtijd']-df2.iloc[i]['Aansluittijd'])/3600), battery - energie_update)
        energie_update = energie_update + bijladen_update
        bijladen.append(bijladen_update)
        verbruik.append(verbruik_update)
        energy.append(energie_update)
    return_df = pd.DataFrame({'energie' : energy[:-1],
                              'verbruik': verbruik,
                             'bijladen' : bijladen,
                             'bijladen_snel' : bijladen_snel,
							 'index' : df2['index']}, index = df2.index)

    return return_df

def bijladen_spread(bijladen, laadvermogen, n_hours): 
    laadvermogen = max(laadvermogen, np.ceil(bijladen/n_hours)+1)
    a = ([laadvermogen]*int(bijladen/laadvermogen)) + [bijladen % laadvermogen]
    a += [0] * (n_hours - len(a))
    return a
	
def bijladen_spread_smart(bijladen, laadvermogen, n_hours): 
    a = [bijladen/n_hours]*n_hours
    return a
	
@st.cache_data
def charge_hour(df, laadvermogen = 44, laadvermogen_snel = 150, aansluittijd = 600, battery = 540, smart = 0):
    df_bijladen = df.loc[df.bijladen > 0].copy()
    df_bijladen['StartTime'] = df_bijladen.apply(lambda row: list(pd.date_range(start = row['Begindatum en -tijd'], 
                                                        end = row['Einddatum en -tijd'], 
                                                        freq = '1h')), axis =1)
        
    df_hour = df_bijladen.explode(column = 'StartTime').reset_index()[['index','StartTime','bijladen','thuis']]
    
    if smart == 0:
        df_hour['bijladen'] = df_hour.groupby('index').bijladen.transform(lambda row: bijladen_spread(row.max(),laadvermogen,len(row)))
    elif smart == 1:
        df_hour['bijladen'] = df_hour.groupby('index').bijladen.transform(lambda row: bijladen_spread_smart(row.max(),laadvermogen,len(row)))
    df_hour['hour'] = df_hour['StartTime'].dt.hour
    df_hour['Date'] = df_hour['StartTime'].dt.date
	
    return df_hour

def check_file(file):
	
    xl = pd.ExcelFile(file)
    sheetnames = xl.sheet_names  # see all sheet names

    # Check if the DataFrame has the required column name
    if sorted(sheetnames) != ['laadlocaties', 'laden', 'parameters', 'ritten']:
        error_message = 'het inputbestand moet sheets bevatten met de namen "ritten", "laden", "parameters" en "laadlocaties". Gebruik het template als voorbeeld.'
        st.error(error_message)
        st.stop()
		
    elif list(pd.read_excel(file, sheet_name = 'laden').columns) != ['Activiteit']:
        error_message = 'De sheet "laden" moet alleen een kolom "Activiteit" bevatten met de activiteiten waarvoor laden is toegestaan (zie nieuw template).'
        st.error(error_message)
        st.stop()
			

def get_params(file):
    
    df_params = pd.read_excel(file, sheet_name = 'parameters').set_index('naam')
	
    battery = df_params.loc['accu'].waarde
    zuinig = df_params.loc['efficiency'].waarde
    aansluittijd = df_params.loc['aansluittijd'].waarde
    laadvermogen = df_params.loc['laadvermogen bedrijf'].waarde
    laadvermogen_snel = df_params.loc['laadvermogen snelweg'].waarde
	
    return battery, zuinig, aansluittijd, laadvermogen, laadvermogen_snel

@st.cache_data
def process_excel_file(file, battery, zuinig, aansluittijd, laadvermogen, laadvermogen_snel, nachtladen, activiteitenladen, snelwegladen):
    # Read the Excel file into a DataFrame
    df = pd.read_excel(file, sheet_name = 'ritten')
    df['Positie'] = df['Positie'].fillna('onbekend')
    df['Afstand'] = df['Afstand'].fillna(0)
	
    optional_columns = df.columns.difference(['Voertuig','Begindatum en -tijd','Einddatum en -tijd','Positie', 'Afstand', 'Activiteit'])
	
    # data cleaning
    df["Begindatum en -tijd"] = pd.to_datetime(df['Begindatum en -tijd'])
    df["Einddatum en -tijd"] = pd.to_datetime(df['Einddatum en -tijd'])

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
    agg_dict = {'Activiteit' : 'first',
                'Positie' : 'first',
                'Afstand' : 'sum',
                'Begindatum en -tijd' : 'min',
                'Einddatum en -tijd' : 'max',
                'Laden' : 'first'}
    
    optional_dict = {key : 'first' for key in optional_columns}
    agg_dict = {**agg_dict, **optional_dict}
	
    df = df.groupby(['Voertuig','activiteit_g']).agg(agg_dict).reset_index(drop = False).drop('activiteit_g', axis = 1)
    df['Activiteit_id'] = df.index
    
    # Prijzen inladen
    # Laad prijzen in vanuit Excelbestand in repository
    prices_path = 'day_ahead_prices.xlsx'
    prices = pd.read_excel(prices_path, engine='openpyxl')
    
    # Maak relevante kolommen aan voor prices tabel
    prices = prices[['datetime_CET', 'price_eur_mwh']].sort_values(by = 'datetime_CET')
    prices['datetime_CET_end'] = prices['datetime_CET'].shift(-1)
    prices['Datum'] = prices['datetime_CET'].dt.date
    
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
                    prices = pd.concat([prices,a])
                # Wanneer API fout geeft, throw exception
                except:
                    print(f'No day ahead prices available for {i}')
    else:
        # Werk met een vaste elektriciteitsprijs, gegeven door het bedrijf of een standaardwaarde
        print('fixed_prices')
    
    # Prijzen mergen aan het dataframe
    df = add_day_ahead_prices(df, prices)

    df['Duur'] = (df['Einddatum en -tijd'] - df['Begindatum en -tijd']).apply(lambda x: x.total_seconds())
    df['nacht'] = np.where(((df.Afstand < 3) & (df.Duur > 6*3600)),1,0)

    if df.Voertuig.nunique() == 1: 
        df['RitID'] = (df['nacht'] < df.shift().fillna(method='bfill')['nacht']).cumsum()
    else: 
        df['RitID'] = df.groupby('Voertuig')['nacht'].transform(lambda g: (g < g.shift().fillna(method='bfill')).cumsum())

    df_locatie = pd.read_excel(file, sheet_name = 'laadlocaties').assign(thuis = 1)
    df = df.merge(df_locatie, how = 'left', on = 'Positie')

    df['thuis'] = df.thuis.fillna(0)
    df = df.sort_values(['Voertuig', 'Begindatum en -tijd']).reset_index()

    if snelwegladen == 0: 
    	df_results = (df.
    			groupby(['Voertuig']).
    			apply(lambda g: simulate(g, battery = battery, zuinig = zuinig, aansluittijd = aansluittijd, laadvermogen = laadvermogen, nachtladen = nachtladen, activiteitenladen = activiteitenladen)))
    elif snelwegladen == 1:
    	df_results = (df.
    			groupby(['Voertuig', 'RitID']).
    			apply(lambda g: simulate2(g, battery = battery, zuinig = zuinig, aansluittijd = aansluittijd, laadvermogen = laadvermogen, nachtladen = nachtladen, activiteitenladen = activiteitenladen)))
    

    df = df.merge(df_results, on = 'index', how = 'left')
    
    # Groepeer op activiteit_id om gesplitste rijen voor uur/kwartierprijzen terug te brengen naar 1 activiteit
    df = aggregate_hourly_costs(df)
    
    df['vertraging'] = np.where(df['bijladen_snel'] > 0, aansluittijd + (3600*df['bijladen_snel']/laadvermogen_snel),0)

    # Voeg een extra regel toe voor ieder voertuig wanneer extra bijladen nodig is
    df = df.groupby('Voertuig').apply(lambda g: bijladen_einde_rit(g, prices, laadvermogen = laadvermogen, battery = battery, aansluittijd = aansluittijd), include_groups = False)
    df = df.reset_index(level=1, drop=True).reset_index()

    df = df.drop('index', axis = 1)
    
    df = df[['Voertuig', 'Activiteit', 'Datum', 'Begindatum en -tijd', 'Einddatum en -tijd', 'Positie', 'Afstand', 'Laden', 'Duur', 'nacht', 'RitID', 'thuis', 'energie', 'verbruik', 'bijladen', 'bijladen_snel', 'Laadkosten (EUR)', 'Gemiddelde laadprijs (EUR/kWh)', 'vertraging']]
    
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
    st.subheader('Tabel met hoeveelheid geladen energie per locatie (top 10)')
    bijladen = df.groupby('Positie').bijladen.sum().reset_index()
    bijladen = pd.concat([bijladen,pd.DataFrame({'Positie' : ['snelweg'],
	    'bijladen' : [df.bijladen_snel.sum()]})]).sort_values(by = 'bijladen', ascending = False).rename(columns = {'bijladen': 'Hoeveelheid energie geladen (kWu)'})
    st.table(bijladen.reset_index(drop = True).loc[lambda d: d['Hoeveelheid energie geladen (kWu)'] >0].head(10))


def plot_demand(df, battery, zuinig, aansluittijd, laadvermogen, laadvermogen_snel):
    
    st.subheader('De gemiddelde verdeling van de energievraag over de dag')
    charge_locations = df.loc[lambda d: d.bijladen >0].groupby('Positie').bijladen.sum().sort_values(ascending = False).index
    filter_option = st.selectbox('Selecteer een locatie', charge_locations, index = 0)
    	
    @st.cache_data
    def filter_data(df, selected_option, smart = 0):

        df_plot = df.loc[df.Positie == selected_option]
        plot_data1 = (charge_hour(df_plot, smart = smart, battery = battery, aansluittijd = aansluittijd, laadvermogen = laadvermogen).groupby(['hour','Date']).bijladen.sum()).reset_index()
        
        uniques = [range(1,24), pd.date_range(plot_data1.Date.min(),plot_data1.Date.max(), freq = '1d')]
        df_hour_date = pd.DataFrame(product(*uniques), columns = ['hour','Date'])
        df_hour_date['Date'] = df_hour_date['Date'].dt.date
		
        plot_data1 = df_hour_date.merge(plot_data1, how = 'left', on = ['hour','Date']).fillna(0)
		
        return plot_data1
		
    percentile_choice = st.radio('Percentiel waarde voor error bar',[75,85,95])
	
	# Create a demand plot
    fig, (ax1) = plt.subplots(1, 1, figsize=(12, 6))

    sns.lineplot(data = filter_data(df, filter_option,0), x = 'hour',y = 'bijladen', errorbar=('pi',percentile_choice), ax = ax1, label = 'regulier laden')
    #plot_data1.rename(columns = {'bijladen' : 'zonder smart charging'}).plot(ax = ax1)

    sns.lineplot(data = filter_data(df, filter_option,1), x = 'hour',y = 'bijladen', errorbar=('pi',percentile_choice), ax = ax1, color = 'red', label = 'smart charging')
    ax1.set_ylabel('Totale energievraag transport (kW)')
    ax1.set_xlabel('Uur van de dag')
    ax1.set_ylim(bottom=-0.5)
    plt.legend()
    plt.tight_layout()
	
    st.pyplot(fig)
	# Offer the file download
	
    if st.button('Download data'):
		
        # Create a BytesIO object
        excel_data = BytesIO()

    # Save the DataFrame to BytesIO as an Excel file
        with pd.ExcelWriter(excel_data, engine='xlsxwriter') as writer:
            for location in charge_locations:
                temp = (charge_hour(df.loc[df.Positie == location], smart = 0, battery = battery, aansluittijd = aansluittijd, laadvermogen = laadvermogen).groupby('hour').bijladen.sum()/n_days)
                temp = df_hour_24h.merge(temp, how = 'left', left_on = 'hour', right_index = True).fillna(0).set_index('hour')
                temp.to_excel(writer, index=True, sheet_name = location[:31])

        # Set the BytesIO object's position to the start
        excel_data.seek(0)
    
        # Create a BytesIO object
        excel_data2 = BytesIO()
    
        # Save the DataFrame to BytesIO as an Excel file
        with pd.ExcelWriter(excel_data2, engine='xlsxwriter') as writer:
            for location in charge_locations:
                temp = (charge_hour(df.loc[df.Positie == location], smart = 1, battery = battery, aansluittijd = aansluittijd, laadvermogen = laadvermogen).groupby('hour').bijladen.sum()/n_days)
                temp = df_hour_24h.merge(temp, how = 'left', left_on = 'hour', right_index = True).fillna(0).set_index('hour')
                temp.to_excel(writer, index=True, sheet_name = location[:31])

        # Set the BytesIO object's position to the start
        excel_data2.seek(0)
	
	    # Set up a style to display buttons side by side
        button_style = """
            <style>
                .side-by-side {
                    display: flex;
                    justify-content: space-between;
                }
            </style>
        """
        st.write(button_style, unsafe_allow_html=True)
    	
        col1, col2 = st.columns(2)
        with col1:
            st.download_button('Download de data voor regulier laden', excel_data, file_name='data_demand_plot.xlsx')
        with col2:
            st.download_button('Download de data voor smart charging', excel_data2, file_name='data_demand_plot_smart.xlsx')


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
    # Template file URL
    template_url = 'https://github.com/DatalabHvA/laadmodel/raw/main/template.xlsx'

    # Request the template file
    response = requests.get(template_url)

    # Create a BytesIO object
    template_data = BytesIO(response.content)

    # Offer the file download
    st.download_button('Download Template', template_data, file_name='template.xlsx')


def main():
    st.title('Laadmodel ZEC')
    st.write("De resultaten van deze tool zijn informatief.  \nDe verstrekte informatie kan onvolledig of niet geheel juist zijn.  \nAan de resultaten van deze tool kunnen geen rechten worden ontleend.")

    # Download template button
    download_template()

    # File upload
    uploaded_file = st.file_uploader('Upload Excelbestand met rittendata', type=['xlsx'])

    nachtladen = st.checkbox('Altijd opladen tijdens overnachting op alle locaties')
    activiteitenladen = st.checkbox('Ook opladen tijdens geselecteerde activiteiten')
    snelwegladen = st.checkbox('Extra snelladen toestaan langs de snelweg')

    if uploaded_file is not None:
        try:
            check_file(uploaded_file)
            battery, zuinig, aansluittijd, laadvermogen, laadvermogen_snel = get_params(uploaded_file)
            df = process_excel_file(uploaded_file, battery = battery, zuinig = zuinig, aansluittijd = aansluittijd, laadvermogen = laadvermogen, laadvermogen_snel = laadvermogen_snel, nachtladen = nachtladen, activiteitenladen = activiteitenladen, snelwegladen = snelwegladen)
            st.header('Modelresultaten:')
            show_haalbaarheid(df)
            show_demand_table(df)            
            plot_demand(df, battery = battery, zuinig = zuinig, aansluittijd = aansluittijd, laadvermogen = laadvermogen, laadvermogen_snel = laadvermogen_snel)
            st.subheader('De eerste 15 regels van het outputbestand')
            st.dataframe(df.head(15))
            download_excel(df)
        except Exception as e:
            st.error(f'Error processing the file: {e}')


if __name__ == '__main__':
    main()
