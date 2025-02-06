import os
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
import zipfile
import tempfile
import streamlit as st
from dotenv import load_dotenv
import hashlib
import logging
import time
from queue import Queue
import pandas as pd
from io import BytesIO
from PyPDF2 import PdfReader
import requests
from pdf_downloader import download_pdf

# Load environment variables
load_dotenv()

# Set API keys from environment variables
groq_api_key = os.getenv("GROQ_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_base_url = "https://api.sree.shop/v1/"

# Set Google API key in environment
os.environ["GOOGLE_API_KEY"] = google_api_key

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize embeddings with error handling
def initialize_embeddings():
    try:
        return GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            task_type="retrieval_document",
            title="resume_analysis",
        )
    except Exception as e:
        st.error(f"Error initializing embeddings: {str(e)}")
        logging.error(f"Embeddings initialization error: {str(e)}")
        return None

# Initialize embeddings
embeddings = initialize_embeddings()

# Get available models
def get_available_models(provider):
    if provider == "Groq":
        return [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768"
        ]
    elif provider == "OpenAI":
        return [
            "claude-3-5-sonnet-20240620",
            "claude-3-5-sonnet",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            "gpt-4o",
            "gpt-4o-2024-05-13"
        ]
    return []

# Initialize LLM (minimal version)
def initialize_llm(provider, model_name):
    if provider == "Groq":
        return ChatGroq(
            model_name=model_name,
            groq_api_key=groq_api_key
        )
    elif provider == "OpenAI":
        return ChatOpenAI(
            model_name=model_name,
            openai_api_key=openai_api_key,
            base_url=openai_base_url
        )

# Process ZIP file containing resumes
def process_zip_file(zip_file):
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        documents = []
        seen_hashes = set()
        for file in os.listdir(temp_dir):
            if file.endswith('.pdf'):
                pdf_path = os.path.join(temp_dir, file)
                file_hash = hashlib.md5(open(pdf_path, "rb").read()).hexdigest()
                if file_hash not in seen_hashes:
                    seen_hashes.add(file_hash)
                    loader = PyPDFLoader(pdf_path)
                    documents.extend(loader.load())
        return documents

# Split documents into chunks
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3500,
        chunk_overlap=700,
        # separators=[]
    )
    return text_splitter.split_documents(documents)

# Create vector store
def create_vectorstore(splits):
    if not splits:
        st.error("No documents to process")
        return None
    try:
        # Try to load existing store
        vectorstore = Chroma(
            persist_directory="chroma_db",
            embedding_function=embeddings
        )
        # Add new documents if any
        if splits:
            vectorstore.add_documents(documents=splits)
        return vectorstore
    except Exception as e:
        logging.error(f"Error with existing vector store: {str(e)}")
        try:
            # Create new store if loading fails
            return Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                persist_directory="chroma_db"
            )
        except Exception as e:
            logging.error(f"Error creating new vector store: {str(e)}")
            st.error("Failed to create vector store")
            return None

# System prompt for LLM
def get_system_prompt():
    return """You are an experienced HR professional with a deep understanding of recruitment, talent acquisition, and performance evaluation. Your primary role is to analyze candidate information and provide insightful feedback based on job descriptions and candidate documents.

**Analysis Framework:**
1. Evaluate technical skills and experience match
2. Consider cultural fit and soft skills
3. Analyze career progression
4. Assess education and certifications
5. Calculate match percentage based on job requirements
6. Identify potential red flags

**Instructions:**
1. Input Analysis:
   * Analyze the provided resume content thoroughly
   * Compare against the provided job description
   * Consider both technical and soft skills alignment

2. Evaluation Criteria:
   * Match core requirements from job description
   * Verify education/experience minimums
   * Evaluate career progression
   * Consider cultural fit indicators

3. Output Format:
   **Overall Assessment:**
   - Suitability Rating: [Highly Suitable/Suitable/Potentially Suitable/Not Suitable]
   - Match Percentage: [X%]

   **Detailed Analysis:**
   - Technical Skills Match: [Details with specific examples]
   - Experience Alignment: [Analysis with timeline]
   - Education & Certifications: [Verification against requirements]
   - Career Progression: [Pattern analysis]
   - Soft Skills & Cultural Fit: [Observations]

   **Scoring Table:**
   | Category | Score | Notes |
   |----------|--------|-------|
   | Technical Skills | X/10 | [Brief justification] |
   | Experience | X/10 | [Brief justification] |
   | Education | X/10 | [Brief justification] |
   | Cultural Fit | X/10 | [Brief justification] |

   **Strengths & Weaknesses:**
   - Key Strengths: [Bullet points with specific examples]
   - Areas for Improvement: [Constructive feedback]

   **Final Recommendation:**
   - Decision: [Proceed/Hold/Reject]
   - Next Steps: [Specific recommendations]
   - Additional Notes: [Any relevant observations]

Always format responses in clear, structured markdown with proper tables and sections.
Be objective and professional in assessments."""

# Create conversation chain
def create_conversation_chain(vectorstore, llm):
    prompt = ChatPromptTemplate.from_messages([
        ("system", get_system_prompt()),
        ("human", """Job Description:
        {context}
        
        Question/Analysis Request: {question}
        
        Please provide a detailed analysis following the specified format and evaluation criteria.""")
    ])
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True,
        verbose=True,
        combine_docs_chain_kwargs={
            "prompt": prompt,
            "document_variable_name": "context"
        },
        memory=None,
        get_chat_history=lambda h: []
    )

# Streamlit UI
st.set_page_config(page_title="HR Resume Analysis", layout="wide")

# Create tabs
tab1, tab2 = st.tabs(["Resume Analysis", "PDF Downloader"])

with tab1:
    # Sidebar
    with st.sidebar:
        st.title("Document Processing")
        uploaded_file = st.file_uploader("Upload ZIP file containing PDFs", type="zip", key="zip_uploader")
        provider = st.selectbox("Select Provider", ["OpenAI", "Groq"])
        model_name = st.selectbox("Select Model", get_available_models(provider))
        process_button = st.button("Process Documents")

        # Preset queries section
        st.markdown("### Quick Analysis")
        preset_queries = {
            "Get Top 15 Candidates": """Analyze all resumes and provide top 15 candidates ranked by match percentage. 
            Format as a markdown table with columns: Rank, Name, Match %, Key Strengths, Recommendation""",
            "Show 5 Best Profiles": """Show detailed analysis of the 5 strongest candidates. 
            Include their key achievements, experience, and why they stand out.""",
            "Best Candidate Analysis": """Identify the single best candidate and provide comprehensive analysis including:
            1. Why they're the best fit
            2. Key qualifications
            3. Experience highlights
            4. Potential growth areas
            5. Recommended interview questions""",
            "Technical Skills Analysis": """Generate a detailed analysis of technical skills across all candidates with:
            | Name | Technical Skills | Experience Level | Skill Match % | Notes |"""
        }
        for query_name, query_text in preset_queries.items():
            if st.button(query_name):
                st.session_state.preset_query = query_text

    # Main chat interface
    st.title("HR Resume Analysis Assistant")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chain" not in st.session_state:
        st.session_state.chain = None
    if "job_description" not in st.session_state:
        st.session_state.job_description = ""
    if "request_queue" not in st.session_state:
        st.session_state.request_queue = Queue()
    if "last_request_time" not in st.session_state:
        st.session_state.last_request_time = 0

    # Add job description input
    if not st.session_state.job_description:
        with st.expander("📋 Enter Job Description", expanded=True):
            job_description = st.text_area(
                "Enter the job description to match candidates against:",
                height=200
            )
            if st.button("Save Job Description"):
                st.session_state.job_description = job_description
                st.success("Job description saved! You can now analyze candidates.")

    # Process documents when button is clicked
    if process_button and uploaded_file and st.session_state.job_description:
        with st.spinner("Processing resumes..."):
            try:
                if embeddings is None:
                    st.error("Failed to initialize embeddings. Please check your Google API key.")
                    logging.error("Embeddings not initialized")
                    st.stop()
                logging.info("Embeddings initialized successfully")
                llm = initialize_llm(provider, model_name)
                documents = process_zip_file(uploaded_file)
                if not documents:
                    st.error("No documents found in the ZIP file")
                    st.stop()
                splits = split_documents(documents)
                vectorstore = create_vectorstore(splits)
                if vectorstore is None:
                    st.error("Failed to create vector store")
                    st.stop()
                st.session_state.chain = create_conversation_chain(vectorstore, llm)
                st.success("Resumes processed successfully!")
            except Exception as e:
                logging.error(f"Error processing documents: {str(e)}")
                st.error(f"Error processing documents: {str(e)}")
    elif process_button and not st.session_state.job_description:
        st.warning("Please enter a job description before processing resumes.")

    # Display chat interface
    if st.session_state.chain:
        # Handle preset query if selected
        if hasattr(st.session_state, 'preset_query'):
            analysis_prompt = f"""Based on this job description:
            {st.session_state.job_description}
            {st.session_state.preset_query}"""
            with st.chat_message("assistant"):
                try:
                    with st.spinner("Analyzing... (This may take a moment due to rate limits)"):
                        response = st.session_state.chain.invoke({
                            "question": analysis_prompt,
                            "chat_history": []
                        })
                    st.markdown(response["answer"])
                    st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
                except Exception as e:
                    if "429" in str(e):
                        st.error("Rate limit reached. Please wait 60 seconds before trying again.")
                    else:
                        st.error(f"An error occurred: {str(e)}")
                del st.session_state.preset_query

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask specific questions about candidates"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                try:
                    with st.spinner("Analyzing... (This may take a moment due to rate limits)"):
                        analysis_prompt = f"""Analyze candidates based on this job description:
                        {st.session_state.job_description}
                        User question: {prompt}
                        Provide analysis in a clear, structured format with relevant details and recommendations."""
                        response = st.session_state.chain.invoke({
                            "question": analysis_prompt,
                            "chat_history": [(msg["role"], msg["content"]) for msg in st.session_state.messages]
                        })
                        st.markdown(response["answer"])
                        st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
                        
                        # Move the source documents section inside the try block
                        if "source_documents" in response and response["source_documents"]:
                            with st.expander("View Source Resumes"):
                                for i, doc in enumerate(response["source_documents"], 1):
                                    st.markdown(f"**Resume {i}:**")
                                    st.markdown(f"```\n{doc.page_content[:200]}...\n```")
                            
                except Exception as e:
                    if "429" in str(e):
                        st.error("Rate limit reached. Please wait 60 seconds before trying again.")
                    else:
                        st.error(f"An error occurred: {str(e)}")

            # Display sources in expander
            if response["source_documents"]:
                with st.expander("View Source Resumes"):
                    for i, doc in enumerate(response["source_documents"], 1):
                        st.markdown(f"**Resume {i}:**")
                        st.markdown(f"```\n{doc.page_content[:200]}...\n```")
        else:
            st.info("Please upload resumes, enter job description, and process documents to start analysis.")

    # Add control buttons
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Clear Chat"):
            st.session_state.messages = []
    with col2:
        if st.button("Clear All"):
            st.session_state.messages = []
            st.session_state.job_description = ""
            st.session_state.chain = None
            st.experimental_rerun()

    # Add before making the API call
    def can_make_request():
        current_time = time.time()
        if current_time - st.session_state.last_request_time < 1:  # 1 second delay between requests
            return False
        return True

with tab2:
    st.title("Excel PDF Downloader")
    
    # File upload for Excel
    excel_file = st.file_uploader("Upload your Excel file", type=['xlsx', 'xls', 'csv'], key="excel_uploader")
    
    if excel_file is not None:
        file_extension = excel_file.name.split('.')[-1]
        
        try:
            if file_extension == 'csv':
                df = pd.read_csv(excel_file)
            elif file_extension == 'xlsx':
                df = pd.read_excel(excel_file, engine='openpyxl')
            elif file_extension == 'xls':
                df = pd.read_excel(excel_file, engine='xlrd')
                
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
                    
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

# Add this function before the tab2 code
def download_pdf(url, filename, retries=3):
    attempt = 0
    while attempt < retries:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                with open(filename, 'wb') as f:
                    f.write(response.content)
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
        time.sleep(2)
    return False