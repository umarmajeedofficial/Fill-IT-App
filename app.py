import streamlit as st
import os
import requests
import pdfplumber
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from io import BytesIO

# Define paths and setup models
device = "cuda:0" if torch.cuda.is_available() else "cpu"
whisper_model_id = "openai/whisper-medium"
whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(whisper_model_id)
whisper_processor = AutoProcessor.from_pretrained(whisper_model_id)
whisper_pipe = pipeline(
    "automatic-speech-recognition",
    model=whisper_model,
    tokenizer=whisper_processor.tokenizer,
    feature_extractor=whisper_processor.feature_extractor,
    device=device
)

granite_url = "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29"
granite_headers = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": "Bearer eyJraWQiOiIyMDI0MDgwMzA4NDEiLCJhbGciOiJSUzI1NiJ9.eyJpYW1faWQiOiJJQk1pZC02OTQwMDBJTlNIIiwiaWQiOiJJQk1pZC02OTQwMDBJTlNIIiwicmVhbG1pZCI6IklCTWlkIiwianRpIjoiODdkNzc1NWUtNzU4Ny00Nzc0LWI4NzAtZjkyNGQ3MGIxNmEzIiwiaWRlbnRpZmllciI6IjY5NDAwMElOU0giLCJnaXZlbl9uYW1lIjoiVW1hciIsImZhbWlseV9uYW1lIjoiTWFqZWVkIiwibmFtZSI6IlVtYXIgTWFqZWVkIiwiZW1haWwiOiJ1bWFybWFqZWVkb2ZmaWNpYWxAZ21haWwuY29tIiwic3ViIjoidW1hcm1hamVlZG9mZmljaWFsQGdtYWlsLmNvbSIsImF1dGhuIjp7InN1YiI6InVtYXJtYWplZWRvZmZpY2lhbEBnbWFpbC5jb20iLCJpYW1faWQiOiJJQk1pZC02OTQwMDBJTlNIIiwibmFtZSI6IlVtYXIgTWFqZWVkIiwiZ2l2ZW5fbmFtZSI6IlVtYXIiLCJmYW1pbHlfbmFtZSI6Ik1hamVlZCIsImVtYWlsIjoidW1hcm1hamVlZG9mZmljaWFsQGdtYWlsLmNvbSJ9LCJhY2NvdW50Ijp7InZhbGlkIjp0cnVlLCJic3MiOiIyZTY5MjI1ZjNmMjc0Nzc2ODkwMGE2MGQ5MDBkM2UzNyIsImltc191c2VyX2lkIjoiMTI2MjI5MTciLCJmcm96ZW4iOnRydWUsImltcyI6IjI3NDQzNDQifSwiaWF0IjoxNzI0Njc4Njc3LCJleHAiOjE3MjQ2ODIyNzcsImlzcyI6Imh0dHBzOi8vaWFtLmNsb3VkLmlibS5jb20vaWRlbnRpdHkiLCJncmFudF90eXBlIjoidXJuOmlibTpwYXJhbXM6b2F1dGg6Z3JhbnQtdHlwZTphcGlrZXkiLCJzY29wZSI6ImlibSBvcGVuaWQiLCJjbGllbnRfaWQiOiJkZWZhdWx0IiwiYWNyIjoxLCJhbXIiOlsicHdkIl19.fmiLcZExa22sN_8Xx3_e-VTvZQVvMqmAi_QiA4NKCV40ni8bobxiFEeBKyv8MpafA405jSzFYQUPRFmBy6XNpvVMWpIYKqsZao7l_EDtqXLDRkM_SySUhZtK4CHu-o6qiLyyObBGabke7niaqXuDhzfvpmZCvA98542aeEwSbYZe6siI9_l05xW1T__fIvKak9Y0Fkf7srAmwW7b0NmezQ0VLH13-hANFm0aXh_sEBT0pGujeyRV6X0Bl0zbNW2YurQzdug23BtdS-IR2xbjoAq9KqsSFK2PUMlA_ENg5oKR00sUqCl3gVvVMRNCFbdSkDnaSv2NWDHH-yhE2LwgTw"  # Replace with your actual API key
}

# Functions
def transcribe_audio(file):
    with BytesIO(file.read()) as audio_file:
        result = whisper_pipe(audio_file)
    return result['text']

def extract_text_from_pdf(file):
    text = ""
    questions = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
                questions += [line.strip() for line in page_text.split("\n") if line.strip()]
    return text, questions

def generate_form_data(text, questions):
    question_list = "\n".join(f"- {question}" for question in questions)
    body = {
        "input": f"""The following text is a transcript from an audio recording. Read the text and extract the information needed to fill out the following form.\n\nText: {text}\n\nForm Questions:\n{question_list}\n\nExtracted Form Data:""",
        "parameters": {
            "decoding_method": "sample",
            "max_new_tokens": 900,
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 1,
            "repetition_penalty": 1.05
        },
        "model_id": "ibm/granite-13b-chat-v2",
        "project_id": "698f0da7-6b34-4642-8540-978e70e85c8e",  # Replace with your actual project ID
        "moderations": {
            "hap": {
                "input": {
                    "enabled": True,
                    "threshold": 0.5,
                    "mask": {"remove_entity_value": True}
                },
                "output": {
                    "enabled": True,
                    "threshold": 0.5,
                    "mask": {"remove_entity_value": True}
                }
            }
        }
    }
    response = requests.post(granite_url, headers=granite_headers, json=body)
    if response.status_code != 200:
        raise Exception("Non-200 response: " + str(response.text))
    data = response.json()
    return data['results'][0]['generated_text'].strip()

def save_responses_to_pdf(responses):
    buffer = BytesIO()
    document = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()

    number_style = ParagraphStyle(
        name='NumberedStyle',
        parent=styles['BodyText'],
        fontSize=10,
        spaceAfter=12
    )

    content = []
    for index, response in enumerate(responses, start=1):
        heading = Paragraph(f"<b>File {index}:</b>", styles['Heading2'])
        response_text = Paragraph(response.replace("\n", "<br/>"), number_style)

        content.append(heading)
        content.append(Spacer(1, 6))
        content.append(response_text)
        content.append(Spacer(1, 18))

    document.build(content)
    buffer.seek(0)
    return buffer

# Streamlit UI
st.title("Audio-to-Text Conversion and Form Filling")

audio_file = st.file_uploader("Upload Audio File", type=["wav", "mp3"])
pdf_file = st.file_uploader("Upload PDF File", type="pdf")

if st.button("Process Files"):
    if audio_file and pdf_file:
        # Transcribe audio
        transcribed_text = transcribe_audio(audio_file)
        
        # Extract text and questions from PDF
        pdf_text, pdf_questions = extract_text_from_pdf(pdf_file)
        
        # Generate form data
        form_data = generate_form_data(transcribed_text, pdf_questions)
        
        # Save responses to PDF
        pdf_buffer = save_responses_to_pdf([form_data])
        
        # Display results
        st.write("Extracted Form Data:")
        st.write(form_data)
        
        # Provide download link for PDF
        st.download_button(
            label="Download Response PDF",
            data=pdf_buffer,
            file_name="response_output.pdf",
            mime="application/pdf"
        )
    else:
        st.warning("Please upload both audio and PDF files.")
