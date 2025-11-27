import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["OPENAI_CA_BUNDLE"] = certifi.where()

import streamlit as st
import base64
import tempfile
import time
import json
import httpx
import pandas as pd
import plotly.express as px
import speech_recognition as sr
from datetime import datetime
from PIL import Image
from io import BytesIO
from fpdf import FPDF

# LangChain Imports
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.messages import HumanMessage

# --- CONFIGURATION & SETUP ---
st.set_page_config(
    page_title="CPG Compliance Guard",
    page_icon="🏷️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Enterprise Look
st.markdown("""
    <style>
    .main { background-color: #f4f6f9; }
    .stButton button { background-color: #0052cc; color: white; border-radius: 6px; font-weight: 600; }
    .metric-card { background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); text-align: center; }
    h1, h2, h3 { color: #172b4d; font-family: 'Segoe UI', sans-serif; }
    .report-container { border: 1px solid #dfe1e6; padding: 20px; border-radius: 8px; background: white; }
    </style>
""", unsafe_allow_html=True)

# --- BACKEND CLASSES ---

class Config:
    """Centralized configuration for GenAI Lab."""
    def __init__(self):
        if 'config' not in st.session_state:
            st.session_state.config = {
                "api_key": "sk-3Hz0hDF3KEjjmgd7-N8nqA",
                "base_url": "https://genailab.tcs.in",
                "llm_model": "azure/genailab-maas-gpt-4o",
                "embed_model": "azure/genailab-maas-text-embedding-3-large"
            }

class RegulationEngine:
    """Handles the 'Rulebook' - Ingests Regulatory PDFs."""
    def __init__(self, config):
        self.http_client = httpx.Client(verify=False)
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=config["api_key"],
            base_url=config["base_url"],
            model=config["embed_model"],
            http_client=self.http_client,
            check_embedding_ctx_length=False
        )
        self.vector_store = None

    def ingest_regulations(self, uploaded_files):
        all_chunks = []
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        
        for file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.read())
                tmp_path = tmp.name
            
            try:
                loader = PyMuPDFLoader(tmp_path)
                docs = loader.load()
                chunks = text_splitter.split_documents(docs)
                all_chunks.extend(chunks)
            finally:
                os.unlink(tmp_path)
        
        if all_chunks:
            self.vector_store = FAISS.from_documents(all_chunks, self.embeddings)
            return True
        return False

    def retrieve_rules(self, query, k=4):
        if not self.vector_store:
            return "No specific regulations uploaded. Using general FDA/EU knowledge."
        retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
        docs = retriever.invoke(query)
        return "\n\n".join([d.page_content for d in docs])

class LabelAnalyzer:
    """Handles Vision Analysis of Product Labels."""
    def __init__(self, config):
        self.http_client = httpx.Client(verify=False)
        self.llm = ChatOpenAI(
            model=config["llm_model"],
            openai_api_key=config["api_key"],
            base_url=config["base_url"],
            http_client=self.http_client,
            temperature=0
        )

    def analyze_label(self, image_b64, regulation_context):
        prompt = f"""
        You are an expert CPG Compliance Officer. 
        Your task is to audit the provided product label image against the following regulatory context.
        
        REGULATORY CONTEXT (Rules to apply):
        {regulation_context}
        
        INSTRUCTIONS:
        1. Extract all text, ingredients, and nutritional data from the label image.
        2. Identify the product category (e.g., Food, Supplement, Cosmetic).
        3. Check for mandatory statements, ingredient formatting, prohibited claims, and readability.
        4. Give a compliance score between 0-100 based on severity level.
        5. Return a JSON object with the following structure:
        {{
            "product_name": "Name",
            "compliance_score": 0-100,
            "status": "Compliant/Non-Compliant",
            "findings": [
                {{"severity": "Critical|Major|Minor", "issue": "Description", "regulation": "Clause/Rule", "recommendation": "Fix"}}
            ],
            "missing_elements": ["List of missing items"],
            "summary": "Executive summary of the audit."
        }}
        """
        
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
            ]
        )
        
        try:
            response = self.llm.invoke([message])
            content = response.content.replace("```json", "").replace("```", "").strip()
            return json.loads(content)
        except Exception as e:
            return {"error": str(e)}

    def chat_with_label(self, chat_history, user_question, chat_context):
        """
        Follow-up Q&A logic.
        chat_context now includes both the Visual Findings JSON and Regulatory Text.
        """
        prompt = f"""
        You are a helpful Compliance Assistant using the following context:
        
        === PRODUCT ANALYSIS (Visual Data) ===
        {chat_context['visual_analysis']}
        
        === REGULATORY GUIDELINES (PDF Data) ===
        {chat_context['regulations']}
        
        USER QUESTION: 
        {user_question}
        
        Answer as a Compliance Expert, referencing both the specific product details and the regulations.
        """
        response = self.llm.invoke(prompt)
        return response.content

class ReportGenerator:
    """Generates PDF Audit Reports with enhanced headers."""
    
    @staticmethod
    def clean_text(text):
        """
        Sanitizes text to remove incompatible unicode characters for FPDF (Latin-1).
        Replaces smart quotes, dashes, etc.
        """
        if not isinstance(text, str):
            return str(text)
            
        replacements = {
            '\u2018': "'", '\u2019': "'",  # Smart single quotes
            '\u201c': '"', '\u201d': '"',  # Smart double quotes
            '\u2013': '-', '\u2014': '-',  # Dashes
            '\u2026': '...',               # Ellipsis
            '\u00a0': ' '                  # Non-breaking space
        }
        
        for char, replacement in replacements.items():
            text = text.replace(char, replacement)
            
        # Final safety net: encode to latin-1 ignoring errors, then decode back
        return text.encode('latin-1', 'replace').decode('latin-1')

    @staticmethod
    def create_pdf(analysis_json):
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        
        # --- 1. Header Section ---
        try:
            if os.path.exists("logo.png"):
                pdf.image("logo.png", x=10, y=8, w=25)
        except:
            pass

        pdf.set_xy(120, 10)
        pdf.set_font("Arial", 'B', 8)
        pdf.set_text_color(100, 100, 100)
        pdf.multi_cell(80, 4, txt="Generated by Team Anushandhan\npowering the next gen AI : TEAM 30", align='R')
        
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pdf.set_xy(120, 20)
        pdf.set_font("Arial", '', 8)
        pdf.cell(80, 4, txt=f"Generated on: {current_time}", align='R', ln=True)

        pdf.ln(15)

        # --- 2. Report Title ---
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Arial", 'B', 16)
        
        product_name = ReportGenerator.clean_text(analysis_json.get('product_name', 'Unknown Product'))
        pdf.cell(0, 10, txt=f"Compliance Audit: {product_name}", ln=True, align='C')
        pdf.ln(5)
        
        # --- 3. Score Card ---
        pdf.set_fill_color(240, 240, 240)
        pdf.rect(10, pdf.get_y(), 190, 25, 'F')
        pdf.set_xy(15, pdf.get_y() + 5)
        
        pdf.set_font("Arial", 'B', 12)
        status = ReportGenerator.clean_text(analysis_json.get('status', 'N/A'))
        pdf.cell(90, 10, txt=f"Compliance Score: {analysis_json.get('compliance_score', 0)}/100", ln=False)
        pdf.cell(90, 10, txt=f"Status: {status}", ln=True)
        pdf.ln(15)
        
        # --- 4. Executive Summary ---
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, txt="Executive Summary", ln=True)
        pdf.set_font("Arial", '', 10)
        
        summary = ReportGenerator.clean_text(analysis_json.get('summary', 'No summary available.'))
        pdf.multi_cell(0, 6, txt=summary)
        pdf.ln(8)
        
        # --- 5. Detailed Findings ---
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, txt="Detailed Findings", ln=True)
        pdf.set_font("Arial", '', 10)
        
        for finding in analysis_json.get('findings', []):
            severity = finding.get('severity', 'Minor')
            
            if severity == 'Critical':
                pdf.set_text_color(200, 0, 0)
            elif severity == 'Major':
                pdf.set_text_color(220, 120, 0)
            else:
                pdf.set_text_color(0, 0, 0)
                
            pdf.set_font("Arial", 'B', 10)
            
            # Clean text before writing
            issue = ReportGenerator.clean_text(f"[{severity}] {finding.get('issue', '')}")
            pdf.multi_cell(0, 8, txt=issue)
            
            pdf.set_text_color(50, 50, 50)
            pdf.set_font("Arial", '', 9)
            
            regulation = ReportGenerator.clean_text(f"Regulation: {finding.get('regulation', 'N/A')}")
            recommendation = ReportGenerator.clean_text(f"Recommendation: {finding.get('recommendation', 'N/A')}")
            
            pdf.multi_cell(0, 5, txt=regulation)
            pdf.multi_cell(0, 5, txt=recommendation)
            pdf.ln(3)
            
        return pdf.output(dest='S').encode('latin-1', errors='replace')

class VoiceInput:
    """Handles speech-to-text functionality."""
    @staticmethod
    def listen():
        r = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("Listening... Speak now.")
            try:
                audio = r.listen(source, timeout=5, phrase_time_limit=10)
                text = r.recognize_google(audio)
                return text
            except Exception:
                return None

# --- UI & MAIN APP ---

def main():
    Config()
    if "regulation_engine" not in st.session_state:
        st.session_state.regulation_engine = RegulationEngine(st.session_state.config)
    if "analyzer" not in st.session_state:
        st.session_state.analyzer = LabelAnalyzer(st.session_state.config)
    if "audit_result" not in st.session_state:
        st.session_state.audit_result = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Sidebar
    with st.sidebar:
        st.header("📚 Knowledge Base")
        reg_files = st.file_uploader("Upload Regulations (PDF)", accept_multiple_files=True, type="pdf")
        if reg_files and st.button("Ingest Regulations"):
            with st.spinner("Indexing Regulatory Framework..."):
                success = st.session_state.regulation_engine.ingest_regulations(reg_files)
                if success: st.success("Knowledge Base Updated!")
        
        st.divider()
        st.header("⚙️ Lab Settings")
        st.session_state.config["api_key"] = st.text_input("API Key", value=st.session_state.config["api_key"], type="password")

    # Main Content
    st.title("🏷️ CPG Compliance Guard")

    # 1. Input Section
    tab1, tab2 = st.tabs(["📁 Upload Label", "📸 Camera Capture"])
    
    image_data = None
    with tab1:
        uploaded_file = st.file_uploader("Upload Label Image", type=["jpg", "png", "jpeg"])
        if uploaded_file: image_data = uploaded_file
    with tab2:
        camera_file = st.camera_input("Take a picture")
        if camera_file: image_data = camera_file

    if image_data:
        st.image(image_data, caption="Product Label", width=300)
        
        # 2. Analysis Trigger
        if st.button("🔍 Run Compliance Audit", type="primary"):
            # RESET CHAT ON NEW RUN
            st.session_state.chat_history = []
            
            with st.spinner("Analyzing visual elements & regulations..."):
                # Context Retrieval
                context = st.session_state.regulation_engine.retrieve_rules("labeling requirements ingredients mandatory statements")
                
                # Image Encoding
                if isinstance(image_data, BytesIO): img_bytes = image_data
                else: img_bytes = image_data
                b64_img = base64.b64encode(img_bytes.getvalue()).decode('utf-8')

                # Analysis
                result = st.session_state.analyzer.analyze_label(b64_img, context)
                
                if "error" in result:
                    st.error(f"Analysis Failed: {result['error']}")
                else:
                    st.session_state.audit_result = result
                    st.session_state.current_context = {
                        "visual_analysis": json.dumps(result),
                        "regulations": context
                    }

    # 3. Results Dashboard
    if st.session_state.audit_result:
        res = st.session_state.audit_result
        st.divider()
        
        # Score Cards
        c1, c2, c3 = st.columns(3)
        c1.metric("Compliance Score", f"{res.get('compliance_score')}/100")
        c2.metric("Status", res.get('status'))
        c3.metric("Issues", len(res.get('findings', [])))

        # Findings Table
        st.subheader("📋 Detailed Findings")
        findings = res.get('findings', [])
        if findings:
            df = pd.DataFrame(findings)
            def color_severity(val):
                color = '#ffcccc' if val in ['Critical', 'Major'] else '#ffffff'
                return f'background-color: {color}'
            st.dataframe(df.style.map(color_severity, subset=['severity']), width='stretch')
        else:
            st.success("No non-compliance issues found!")

        # Export
        col_export, _ = st.columns([1, 4])
        with col_export:
            pdf_bytes = ReportGenerator.create_pdf(res)
            st.download_button("📄 Download Audit Report", pdf_bytes, "audit_report.pdf", "application/pdf")

        # 4. Interactive Chat (With Voice)
        st.divider()
        st.subheader("💬 Audit Assistant")

        # Display History
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Input Controls
        col_text, col_mic = st.columns([8, 1])
        
        voice_input = None
        with col_mic:
            # Voice Button
            if st.button("🎤", help="Speak"):
                voice_input = VoiceInput.listen()
        
        with col_text:
            # Standard Chat Input
            text_input = st.chat_input("Ask about the findings...")

        # Determine Input Source
        final_query = voice_input if voice_input else text_input

        if final_query:
            # User Message
            st.session_state.chat_history.append({"role": "user", "content": final_query})
            with st.chat_message("user"):
                st.markdown(final_query)

            # AI Response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer = st.session_state.analyzer.chat_with_label(
                        st.session_state.chat_history, 
                        final_query, 
                        st.session_state.current_context
                    )
                    st.markdown(answer)
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()