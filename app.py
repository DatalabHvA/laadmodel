import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests
from io import BytesIO


def process_excel_file(file):
    # Read the Excel file into a DataFrame
    df = pd.read_excel(file)

    # Add a new column with squared values
    df['Squared'] = df.iloc[:, 0].apply(lambda x: x ** 2)

    return df


def plot_scatter(df):
    # Create a scatter plot
    fig, ax = plt.subplots()
    ax.scatter(df.iloc[:, 0], df['Squared'])
    ax.set_xlabel('Original')
    ax.set_ylabel('Squared')
    ax.set_title('Original vs. Squared')
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
    st.download_button('Download Excel file', excel_data, file_name='modified_file.xlsx')


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
    st.title('Excel File Processor')

    # Download template button
    download_template()

    # File upload
    uploaded_file = st.file_uploader('Upload Excel file', type=['xlsx'])

    if uploaded_file is not None:
        try:
            df = process_excel_file(uploaded_file)
            plot_scatter(df)
            download_excel(df)
        except Exception as e:
            st.error(f'Error processing the file: {e}')


if __name__ == '__main__':
    main()
