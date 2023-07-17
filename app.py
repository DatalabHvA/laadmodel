import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests
from io import BytesIO
import numpy as np

def simulate(df2, zuinig = 1, laadvermogen = 44, laadvermogen_snel = 150, aansluittijd = 600, battery = 300):
    #print(df2.Voertuig.iloc[0])

    df2['Laadtijd'] = np.where((df2['thuis'] == 1) & (df2['Laden'] == 1), df2['Duur'],0) # alleen thuis laden


    energy = [battery]
    bijladen = []
    bijladen_snel = []

    for i in range(df2.shape[0]):
        #print(str(i))
        afstand = df2.iloc[i]['Afstand']
        energie_update = energy[i] - (zuinig*afstand)
        
        bijladen_snel_update = -energie_update if energie_update < 0 else 0
        energie_update = energie_update + bijladen_snel_update
        bijladen_snel.append(bijladen_snel_update)
        
        bijladen_update = min(laadvermogen*(max(0, df2.iloc[i]['Laadtijd']-aansluittijd)/3600), battery - energie_update)
        energie_update = energie_update + bijladen_update
        bijladen.append(bijladen_update)
        energy.append(energie_update)
    return_df = pd.DataFrame({'energie' : energy[:-1],
                             'bijladen' : bijladen,
                             'bijladen_snel' : bijladen_snel}, index = df2.index)

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

def process_excel_file(file):
    # Read the Excel file into a DataFrame
    df = pd.read_excel(file, sheet_name = 'ritten')
    df = df.sort_values(['Voertuig', 'Begindatum en -tijd']).reset_index(drop = True)

    # data cleaning
    df["Begindatum en -tijd"] = pd.to_datetime(df['Begindatum en -tijd'])
    df["Einddatum en -tijd"] = pd.to_datetime(df['Einddatum en -tijd'])
    
    # fill gaps in the time series with 'Rusten' activity
    df['lag'] = df.groupby('Voertuig')['Einddatum en -tijd'].shift()
    mask = (df['Begindatum en -tijd'] !=  df['lag']) & df['lag'].notna()
    
    rows = df.loc[mask].drop(columns=['Einddatum en -tijd'])
	
    rows = rows.rename(columns={'Begindatum en -tijd': 'Einddatum en -tijd', 'lag': 'Begindatum en -tijd'})
    rows['Activiteit'] = 'Rusten'
    rows['Afstand'] = 0
    
    #Realign index to ensure the order while sorting in next step
    rows.index -= 1 
    
    #append the new rows and sort the index
    df = pd.concat([df, rows]).sort_index(ignore_index=True).drop(columns='lag')
    
    df = df.sort_values(['Voertuig', 'Begindatum en -tijd']).reset_index(drop = True)
    
    #set which activities can be used for charging
    df_act = pd.read_excel(file, sheet_name = 'laden')
    df = df.merge(df_act, how = 'left', on = 'Activiteit')
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
	
    df = df.sort_values(['Voertuig', 'Begindatum en -tijd']).reset_index(drop = True)
	
    df_params = pd.read_excel(file, sheet_name = 'parameters').set_index('naam')
	
    df = df.join(df.
    		groupby('Voertuig').
    		apply(lambda g: simulate(g, 
    			battery = df_params.loc['accu'].waarde,
    			zuinig = df_params.loc['efficiency'].waarde,
    			aansluittijd = df_params.loc['aansluittijd'].waarde,
    			laadvermogen = df_params.loc['laadvermogen'].waarde)))

    return df


def plot_scatter(df):
    # Create a scatter plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    n_days = max(1,(df['Begindatum en -tijd'].max()-df['Begindatum en -tijd'].min()).days)

    (charge_hour(df, smart = 0).groupby('hour').bijladen.sum()/n_days).plot(ax = ax1)
    ax1.set_ylabel('Energievraag zonder smart charging (kW)')
    ax1.set_ylim(bottom=-0.5)

    (charge_hour(df, smart = 1).groupby('hour').bijladen.sum()/n_days).plot(ax = ax2)
    ax2.set_ylabel('Energievraag met smart charging (kW)')
    ax2.set_ylim(bottom=-0.5)
	
    plt.tight_layout()

    st.pyplot(fig)


def download_excel(df):
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

    if uploaded_file is not None:
        try:
            df = process_excel_file(uploaded_file)
            plot_scatter(df)
            download_excel(df)
        except Exception as e:
            st.error(f'Error processing the file: {e}')


if __name__ == '__main__':
    main()
