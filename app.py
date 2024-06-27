import streamlit as st
import hydralit_components as hc
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import streamlit as st
import tempfile
from langchain_huggingface import HuggingFaceEmbeddings
import guardrails as gr
from guardrails.hub import ToxicLanguage
import os

azure_config = {
    "base_url": os.getenv('AZURE_OPENAI_ENDPOINT'),
    "model_deployment": "GPT_35_TURBO_DEMO_RAG_DEPLOYMENT_DONO",
    "model_name": "gpt-35-turbo",
    "embedding_deployment": "ADA_RAG_DONO_DEMO",
    "embedding_name": "text-embedding-ada-002",
    "api-key": os.getenv('AZURE_OPENAI_API_KEY'),
    "api_version": "2024-02-01"
    }

model = AzureChatOpenAI(
    temperature=0,
    model= azure_config["model_deployment"],
    api_key= azure_config["api-key"],
    api_version= azure_config["api_version"],
    azure_endpoint= azure_config["base_url"]
    )

embeddings = AzureOpenAIEmbeddings(
    model= azure_config["embedding_deployment"],
    api_key= azure_config["api-key"],
    api_version= azure_config["api_version"],
    azure_endpoint= azure_config["base_url"]
    )

# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def data_loader(file_path):
    pdf_data = {"text": "", "metadata": {}}
    loader = PyMuPDFLoader(file_path)
    data = loader.load()
    pdf_data['metadata'] = data[0].metadata

    for page_num in data:
        page = page_num.page_content
        pdf_data["text"] += page

    return pdf_data


def get_text_chunks(text):
    text_splitter = SemanticChunker(embeddings)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    # vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    vector_store.add_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
        Context:
        {context}
 
        Question:
        {question}
 
        Instructions:
        1. Carefully read the provided context.
        2. Break down the information step-by-step, analyzing each relevant part.
        3. Use logical reasoning to piece together the answer from the context.
        4. Provide a detailed and comprehensive answer based on the context.
        5. If the answer is not found in the context, respond with: "The answer is not available in the context."
        6. Do not fabricate information or provide incorrect answers.
 
        Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])

# specify the primary menu definition
menu_data = [
        {'icon': "far fa-copy", 'label':"Upload Documents"}
]
# we can override any part of the primary colors of the menu
#over_theme = {'txc_inactive': '#FFFFFF','menu_background':'red','txc_active':'yellow','option_active':'blue'}
over_theme = {'txc_inactive': '#FFFFFF'}
menu_id = hc.nav_bar(menu_definition=menu_data,home_name='Home',override_theme=over_theme)

# #Guardrails
# Initialize Guardrails AI with the configuration
full_guard = gr.Guard.from_string(
    validators=[ToxicLanguage(on_fail="fix")],
    description="testmeout",
)

def main():
    # Main page navigation
    st.title("Welcome to SciQueryBot💁")
    st.write("""
        SciQueryBot is a chatbot application designed to help you extract and understand information from scientific research papers. 
        You can also upload more PDF documents, and the bot will help you with any questions you might have regarding the content.
        """)

    # Buttons for navigation
    if 'page' not in st.session_state:
        st.session_state.page = 'Home'

    if menu_id=='Home':
        st.session_state.page = 'Home'
    if menu_id=="Upload Documents":
        st.session_state.page = 'Upload Documents'


    # Display selected page
    if st.session_state.page == 'Home':
        st.title("Ask Questions")
        user_question = st.text_input("Ask a Question from the PDF Files", key="user_question")
        if user_question:  # Ensure user question is provided
            raw_llm_output, validated_output, *rest = full_guard.parse(
                    llm_output=user_question,)
            user_input(validated_output)
    elif st.session_state.page == 'Upload Documents':
        st.title("Upload Documents")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, key="pdf_uploader")
        if st.button("Submit & Process", key="process_button"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    all_text = ""
                    all_metadata = []
                    for uploaded_file in pdf_docs:
                        # Save the uploaded file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                            temp_file.write(uploaded_file.read())
                            temp_file_path = temp_file.name
                        pdf_data = data_loader(temp_file_path)
                        all_text += pdf_data["text"]
                        all_metadata.append(pdf_data["metadata"])

                    text_chunks = get_text_chunks(all_text)
                    get_vector_store(text_chunks)
                    st.success("Documents processed successfully!")
            else:
                st.warning("Please upload at least one PDF file.")


if __name__ == "__main__":
    main()
    