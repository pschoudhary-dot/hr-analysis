import streamlit as st
import pandas as pd
import requests
import os
import zipfile
from io import BytesIO
from PyPDF2 import PdfReader
import time

def download_pdf(url, filename, retries=3):
    """Download a PDF from a URL and save it to a file, with retries on failure"""
    attempt = 0
    while attempt < retries:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                with open(filename, 'wb') as f:
                    f.write(response.content)

                # Validate if the file is a valid PDF
                try:
                    reader = PdfReader(filename)
                    if len(reader.pages) > 0:
                        return True
                    else:
                        st.warning(f"Downloaded PDF from {url} is empty. Skipping...")
                        os.remove(filename)
                        return False
                except Exception as e:
                    st.warning(f"Error reading PDF from {url}: {str(e)}")
                    os.remove(filename)
                    return False
            else:
                st.warning(f"Failed to download PDF from {url}. HTTP status code: {response.status_code}")
        except Exception as e:
            st.warning(f"Error downloading PDF from {url}: {str(e)}")

        attempt += 1
        time.sleep(2)  # Wait before retrying
    return False

def main():
    st.title("Excel PDF Downloader")
    
    # File upload
    uploaded_file = st.file_uploader("Upload your Excel file", type=['xlsx', 'xls', 'csv'])
    
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1]
        
        try:
            if file_extension == 'csv':
                df = pd.read_csv(uploaded_file)
            elif file_extension == 'xlsx':
                df = pd.read_excel(uploaded_file, engine='openpyxl')
            elif file_extension == 'xls':
                df = pd.read_excel(uploaded_file, engine='xlrd')
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            return
        
        # Show preview of the data  
        st.subheader("Preview of uploaded file")
        st.dataframe(df.head())
        
        # Column selection
        columns = df.columns.tolist()
        selected_column = st.selectbox("Select the column containing PDF URLs", columns)
        
        if st.button("Download PDFs"):
            # Create a temporary directory to store PDFs
            if not os.path.exists("temp_pdfs"):
                os.makedirs("temp_pdfs")
            
            # Download PDFs
            st.info("Downloading PDFs... Please wait.")
            progress_bar = st.progress(0)
            
            successful_downloads = 0
            total_urls = len(df[selected_column].dropna())
            
            for index, url in enumerate(df[selected_column].dropna()):
                filename = f"temp_pdfs/file_{index}.pdf"
                if download_pdf(url, filename):
                    successful_downloads += 1
                progress_bar.progress((index + 1) / total_urls)
            
            # Create ZIP file
            if successful_downloads > 0:
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    for file in os.listdir("temp_pdfs"):
                        file_path = os.path.join("temp_pdfs", file)
                        zip_file.write(file_path, file)
                
                # Clean up temporary files
                for file in os.listdir("temp_pdfs"):
                    os.remove(os.path.join("temp_pdfs", file))
                os.rmdir("temp_pdfs")
                
                # Offer ZIP file for download
                st.success(f"Successfully downloaded {successful_downloads} PDFs")
                st.download_button(
                    label="Download ZIP file",
                    data=zip_buffer.getvalue(),
                    file_name="downloaded_pdfs.zip",
                    mime="application/zip"
                )
            else:
                st.error("No PDFs were successfully downloaded")

if __name__ == "__main__":
    main()
