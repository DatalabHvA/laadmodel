import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from  matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import requests
from io import BytesIO
import numpy as np

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

def simulate2(df2, zuinig = 1.25, laadvermogen = 44, laadvermogen_snel = 150, aansluittijd = 600, battery = 300, nachtladen = 0, activiteitenladen = 0):

    if (nachtladen == 0) & (activiteitenladen == 0):
    	df2['Laadtijd'] = np.where((df2['thuis'] == 1), df2['Duur'],0) # alleen thuis laden
    elif (nachtladen == 1) & (activiteitenladen == 0):
    	df2['Laadtijd'] = np.where((df2['thuis'] == 1) | (df2['nacht'] == 1), df2['Duur'],0) # thuis of 's nachts laden
    elif (nachtladen == 0) & (activiteitenladen == 1):
    	df2['Laadtijd'] = np.where((df2['thuis'] == 1) | (df2['Laden'] == 1), df2['Duur'],0) # thuis of tijdens activiteit
    elif (nachtladen == 1) & (activiteitenladen == 1):
    	df2['Laadtijd'] = np.where((df2['thuis'] == 1) | (df2['Laden'] == 1) | (df2['nacht'] == 1), df2['Duur'],0) # thuis, 's nachts of tijdens activiteit 		
		
    df2['Laadtijd'] = np.where((df2['Activiteit'] == 'Rijden') | (df2['Afstand'] >= 3), 0, df2['Laadtijd']) # niet AC-laden tijdens rijden  		
    df2['laad_potentiaal1'] =df2['Laadtijd'].apply(lambda x: min(battery,
                                                           max(0,((x-aansluittijd)/3600))*laadvermogen))
    df2 = df2.merge(tekort_snel(df2, battery = battery, zuinig = zuinig), left_index = True, right_index = True, how = 'left')
    df2['bijladen_snel'] = df2['bijladen_snel'].fillna(0)
    df2['bijladen'] = df2['bijladen'].fillna(0)
   
    energy = [battery]
    bijladen = []
    bijladen_snel = []

    for i in range(df2.shape[0]):
        verbruik = df2.iloc[i]['Afstand']
        energie_update = energy[i] - (zuinig*verbruik)
        
        bijladen_snel_update = df2.iloc[i]['bijladen_snel']
        energie_update = energie_update + bijladen_snel_update
        bijladen_snel.append(bijladen_snel_update)
        
        bijladen_update = df2.iloc[i]['bijladen']
        energie_update = energie_update + bijladen_update
        bijladen.append(bijladen_update)
        
        energy.append(energie_update)
    
    return_df = pd.DataFrame({'energie' : energy[:-1],
                             'bijladen' : bijladen,
                             'bijladen_snel' : bijladen_snel,
    						 'index' : df2['index']}, index = df2.index)
    return return_df

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
    bijladen = []
    bijladen_snel = []
    
    for i in range(df2.shape[0]):
    
        afstand = df2.iloc[i]['Afstand']
        energie_update = energy[i] - (zuinig*afstand)
        
        bijladen_snel_update = 0
        energie_update = energie_update + bijladen_snel_update
        bijladen_snel.append(bijladen_snel_update)
        
        bijladen_update = min(laadvermogen*(max(0, df2.iloc[i]['Laadtijd']-aansluittijd)/3600), battery - energie_update)
        energie_update = energie_update + bijladen_update
        bijladen.append(bijladen_update)
        energy.append(energie_update)
    return_df = pd.DataFrame({'energie' : energy[:-1],
                             'bijladen' : bijladen,
                             'bijladen_snel' : bijladen_snel,
							 'index' : df2['index']}, index = df2.index)

    return return_df

def bijladen_spread(bijladen, laadvermogen, n_hours): 
    a = ([laadvermogen]*int(bijladen/laadvermogen)) + [bijladen % laadvermogen]
    a += [0] * (n_hours - len(a))
    return a
	
def bijladen_spread_smart(bijladen, laadvermogen, n_hours): 
    a = [bijladen/n_hours]*n_hours
    return a
	
def charge_hour(df, laadvermogen = 44, laadvermogen_snel = 150, aansluittijd = 600, battery = 300, smart = 0):
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

def process_excel_file(file, battery, zuinig, aansluittijd, laadvermogen, laadvermogen_snel, nachtladen, activiteitenladen, snelwegladen):
    # Read the Excel file into a DataFrame
    df = pd.read_excel(file, sheet_name = 'ritten')
    df['Positie'] = df['Positie'].fillna('onbekend')
    df = df.sort_values(['Voertuig', 'Begindatum en -tijd']).reset_index(drop = True)

    # data cleaning
    df["Begindatum en -tijd"] = pd.to_datetime(df['Begindatum en -tijd'])
    df["Einddatum en -tijd"] = pd.to_datetime(df['Einddatum en -tijd'])
    
    # fill gaps in the time series with 'Rusten' activity
    df['lag'] = df.groupby('Voertuig')['Einddatum en -tijd'].shift(1)
    mask = (df['Begindatum en -tijd'] !=  df['lag']) & df['lag'].notna()
   
    rows = df.loc[mask].drop(columns=['Einddatum en -tijd'])
    rows = rows.rename(columns={'Begindatum en -tijd': 'Einddatum en -tijd', 'lag': 'Begindatum en -tijd'})
    rows['Activiteit'] = 'Rusten'
    rows['Afstand'] = 0
        
    #append the new rows and sort the index
    
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

    df = df.groupby(['Voertuig','activiteit_g']).agg({'Activiteit' : 'first',
                               'Positie' : 'first',
                               'Afstand' : 'sum',
                               'Begindatum en -tijd' : 'min',
                               'Einddatum en -tijd' : 'max',
                               'Laden' : 'first'}).reset_index(drop = False).drop('activiteit_g', axis = 1)

    df['Duur'] = (df['Einddatum en -tijd'] - df['Begindatum en -tijd']).apply(lambda x: x.total_seconds())
    df['nacht'] = np.where(((df.Activiteit == 'Rusten') & (df.Duur > 6*3600)),1,0)

    if df.Voertuig.nunique() == 1: 
        df['RitID'] = (df['nacht'] < df.shift().fillna(method='bfill')['nacht']).cumsum()
    else: 
	    df['RitID'] = (df.groupby('Voertuig',group_keys=False).
                      apply(lambda g: (g['nacht'] < g.shift().fillna(method='bfill')['nacht']).cumsum()))

    df_locatie = pd.read_excel(file, sheet_name = 'laadlocaties').assign(thuis = 1)
    df = df.merge(df_locatie, how = 'left', on = 'Positie')
    df['thuis'] = df.thuis.fillna(0)
	
    df = df.sort_values(['Voertuig', 'Begindatum en -tijd']).reset_index()
	
    if snelwegladen == 0: 
    	df_results = (df.
    			groupby(['Voertuig', 'RitID']).
    			apply(lambda g: simulate(g, battery = battery, zuinig = zuinig, aansluittijd = aansluittijd, laadvermogen = laadvermogen, nachtladen = nachtladen, activiteitenladen = activiteitenladen)))
    elif snelwegladen == 1:
    	df_results = (df.
    			groupby(['Voertuig', 'RitID']).
    			apply(lambda g: simulate2(g, battery = battery, zuinig = zuinig, aansluittijd = aansluittijd, laadvermogen = laadvermogen, nachtladen = nachtladen, activiteitenladen = activiteitenladen)))
    
    df = df.merge(df_results, on = 'index', how = 'left')
	
    df['vertraging'] = 3600*df['bijladen_snel']/laadvermogen_snel

    return df.drop(['index'], axis = 1)

def show_haalbaarheid(df):
    df['Datum'] = df.groupby(['Voertuig','RitID'])['Begindatum en -tijd'].transform(lambda x: x.min().date())

    haalbaarheid = df.pivot_table(values = 'energie', index = 'Voertuig', columns = ['Datum'], aggfunc=lambda x: 1 if min(x) >= 0 else 0)
    cmap=LinearSegmentedColormap.from_list('rg',["r","y", "g"], N=256) 
    st.subheader('Haalbaarheid van de planning per dag en voertuig')
    fig1 = plt.figure(figsize=(10, 4))
    sns.heatmap(haalbaarheid, annot=True, cmap = cmap, vmin=0, vmax=1, linewidths=1, linecolor='black')
	
    st.pyplot(fig1)
 
def show_demand_table(df):
    st.subheader('Tabel met hoeveelheid geladen energie per locatie')
    bijladen = df.groupby('Positie').bijladen.sum().reset_index()
    bijladen = pd.concat([bijladen,pd.DataFrame({'Positie' : ['snelweg'],
	    'bijladen' : [df.bijladen_snel.sum()]})]).sort_values(by = 'bijladen', ascending = False).rename(columns = {'bijladen': 'Hoeveelheid energie geladen (kWu)'})
    st.table(bijladen.reset_index(drop = True))


def plot_demand(df, battery, zuinig, aansluittijd, laadvermogen, laadvermogen_snel):
    
    filter_options = list(df.loc[lambda d: d.bijladen >0].Positie.unique())
    highest_demand = df.groupby('Positie')['bijladen'].sum().idxmax()
    default_idx = filter_options.index(highest_demand)
    st.subheader('De gemiddelde verdeling van de energievraag over de dag')
    filter_option = st.selectbox('Selecteer een locatie', df.loc[lambda d: d.bijladen >0].Positie.unique(), index = default_idx)

    df_hour_24h = pd.DataFrame({'hour' : range(1,24)})
	
    df_plot = df.loc[df.Positie == filter_option]
	
	# Create a demand plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    n_days = max(1,(df['Begindatum en -tijd'].max()-df['Begindatum en -tijd'].min()).days)

    plot_data1 = (charge_hour(df_plot, smart = 0, battery = battery, aansluittijd = aansluittijd, laadvermogen = laadvermogen).groupby('hour').bijladen.sum()/n_days)
    plot_data1 = df_hour_24h.merge(plot_data1, how = 'left', left_on = 'hour', right_index = True).fillna(0).set_index('hour')
    plot_data1.plot(ax = ax1)
    ax1.set_ylabel('Gemiddelde energievraag zonder smart charging (kW)')
    ax1.set_ylim(bottom=-0.5)

    plot_data2 = (charge_hour(df_plot, smart = 1, battery = battery, aansluittijd = aansluittijd, laadvermogen = laadvermogen).groupby('hour').bijladen.sum()/n_days)
    plot_data2 = df_hour_24h.merge(plot_data2, how = 'left', left_on = 'hour', right_index = True).fillna(0).set_index('hour')
    plot_data2.plot(ax = ax2)
    ax2.set_ylabel('Gemiddelde energievraag met smart charging (kW)')
    ax2.set_ylim(bottom=-0.5)
	
    plt.tight_layout()

    # Create a BytesIO object
    excel_data = BytesIO()

    # Save the DataFrame to BytesIO as an Excel file
    with pd.ExcelWriter(excel_data, engine='xlsxwriter') as writer:
        plot_data1.to_excel(writer, index=True, sheet_name='normal charging')
        plot_data2.to_excel(writer, index=True, sheet_name='smart charging')

    # Set the BytesIO object's position to the start
    excel_data.seek(0)

    # Offer the file download
    st.pyplot(fig)
	
    st.download_button('Download de data van de plot', excel_data, file_name='data_demand_plot.xlsx')


def download_excel(df):

    st.subheader('Definitieve dataset')
	
    df = df.drop(['Datum'], axis = 1)
	
    # Create a BytesIO object
    excel_data = BytesIO()

    # Save the DataFrame to BytesIO as an Excel file
    with pd.ExcelWriter(excel_data, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')

    # Set the BytesIO object's position to the start
    excel_data.seek(0)
	
    # Offer the file download
    st.download_button('Download Excelbestand met modeluitkomsten', excel_data, file_name='laadmodel_resultaten.xlsx')


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
            st.subheader('TEST: eerste 10 regels van de tabel')
            st.dataframe(df.head(15))
            download_excel(df)
        except Exception as e:
            st.error(f'Error processing the file: {e}')


if __name__ == '__main__':
    main()