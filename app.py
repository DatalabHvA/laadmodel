import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from  matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import requests
from io import BytesIO
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
# Simulatie voor het laadmodel MET de optie voor bijladen langs de snelweg
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
        
        bijladen_update = min(laadvermogen*(max(0, df2.iloc[i]['Laadtijd']-aansluittijd)/3600), battery - energie_update)
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
    agg_dict = {'Activiteit' : 'first',
                'Positie' : 'first',
                'Afstand' : 'sum',
                'Begindatum en -tijd' : 'min',
                'Einddatum en -tijd' : 'max',
                'Laden' : 'first'}
    
    optional_dict = {key : 'first' for key in optional_columns}
    agg_dict = {**agg_dict, **optional_dict}
	
    df = df.groupby(['Voertuig','activiteit_g']).agg(agg_dict).reset_index(drop = False).drop('activiteit_g', axis = 1)

    df['Duur'] = (df['Einddatum en -tijd'] - df['Begindatum en -tijd']).apply(lambda x: x.total_seconds())
    df['nacht'] = np.where(((df.Afstand < 3) & (df.Duur > 6*3600)),1,0)

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
    			groupby(['Voertuig']).
    			apply(lambda g: simulate(g, battery = battery, zuinig = zuinig, aansluittijd = aansluittijd, laadvermogen = laadvermogen, nachtladen = nachtladen, activiteitenladen = activiteitenladen)))
    elif snelwegladen == 1:
    	df_results = (df.
    			groupby(['Voertuig', 'RitID']).
    			apply(lambda g: simulate2(g, battery = battery, zuinig = zuinig, aansluittijd = aansluittijd, laadvermogen = laadvermogen, nachtladen = nachtladen, activiteitenladen = activiteitenladen)))
    
    df = df.merge(df_results, on = 'index', how = 'left')
	
    df['vertraging'] = np.where(df['bijladen_snel'] > 0, 600 + (3600*df['bijladen_snel']/laadvermogen_snel),0)

    return df.drop(['index'], axis = 1)

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
            st.subheader('De eerste 15 regels van de het outputbestand')
            st.dataframe(df.head(15))
            download_excel(df)
        except Exception as e:
            st.error(f'Error processing the file: {e}')


if __name__ == '__main__':
    main()
