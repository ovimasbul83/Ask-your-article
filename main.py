from dotenv import load_dotenv
import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import PyPDFLoader

from langchain.chains.question_answering import load_qa_chain
from langchain_community.embeddings import CohereEmbeddings
from langchain_google_genai import GoogleGenerativeAI


def main():
    load_dotenv()
    google_api_key = os.getenv("GOOGLE_API_KEY")
    st.set_page_config(page_title="Ask your Article", page_icon=":shark:", layout="centered",)
    st.header("Ask your Article 📚")

    # upload file
    pdf_file = st.file_uploader("Upload your PDF", type="pdf")
    summaries = []

    # extract the text
    if pdf_file is not None:
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = CharacterTextSplitter(
            separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
        )
        chunks = text_splitter.split_text(text)
        embeddings = CohereEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)


        user_question = st.text_input("Ask a question about your PDF:")
        if user_question:
            if st.button("Ask"):
              docs = knowledge_base.similarity_search(user_question)

              llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=google_api_key)

              chain = load_qa_chain(llm, chain_type="stuff")
              response = chain.run(input_documents=docs, question=user_question)

              st.write(response)
       



      




if __name__ == "__main__":
    main()

