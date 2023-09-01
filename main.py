from functions import *


def main():
    st.title('Laadmodel ZEC')

    # Download template button
    download_template()

    # File upload
    uploaded_file = st.file_uploader(
        'Upload Excelbestand met rittendata', type=['xlsx'])

    nachtladen = st.checkbox(
        'Altijd opladen tijdens overnachting op alle locaties')
    activiteitenladen = st.checkbox(
        'Ook opladen tijdens geselecteerde activiteiten')
    snelwegladen = st.checkbox('Extra snelladen toestaan langs de snelweg')

    if uploaded_file is not None:
        try:
            check_file(uploaded_file)
            battery, zuinig, aansluittijd, laadvermogen, laadvermogen_snel = get_params(
                uploaded_file)
            df = process_excel_file(uploaded_file, battery=battery, zuinig=zuinig, aansluittijd=aansluittijd, laadvermogen=laadvermogen,
                                    laadvermogen_snel=laadvermogen_snel, nachtladen=nachtladen, activiteitenladen=activiteitenladen, snelwegladen=snelwegladen)
            st.header('Modelresultaten:')
            show_haalbaarheid(df)
            show_demand_table(df)
            plot_demand(df, battery=battery, zuinig=zuinig, aansluittijd=aansluittijd,
                        laadvermogen=laadvermogen, laadvermogen_snel=laadvermogen_snel)
            st.subheader('De eerste 15 regels van de het outputbestand')
            st.dataframe(df.head(15))
            download_excel(df)
        except Exception as e:
            st.error(f'Error processing the file: {e}')


if __name__ == '__main__':
    main()
