#from vertexai.preview.language_models import TextGenerationModel
import vertexai
import streamlit as st
import os
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.embeddings import VertexAIEmbeddings
from langchain.llms import VertexAI 
from langchain.vectorstores import MatchingEngine
from google.cloud import aiplatform
from google.cloud import storage

PROJECT_ID = os.environ.get('GCP_PROJECT') #Your Google Cloud Project ID
LOCATION = os.environ.get('GCP_REGION')   #Your Google Cloud Project Region
vertexai.init(project=PROJECT_ID, location=LOCATION)
GCS_BUCKET_NAME = f'{PROJECT_ID}-{DISPLAY_NAME}'
EMBEDDING_MODEL = VertexAIEmbeddings(model_name="textembedding-gecko")
matching_engine_index = aiplatform.MatchingEngineIndex(
    INDEX_ID
)

matching_engine_endpoint = aiplatform.MatchingEngineIndexEndpoint(
    ENDPOINT_ID,
)

@st.cache_resource
def get_model():
    #generation_model = TextGenerationModel.from_pretrained("text-bison@001")
    generation_model = VertexAI(model_name="code-bison-32k", max_output_tokens=8092, temperature=.2)
    return generation_model

def get_text_generation(prompt="",  **parameters):
    generation_model = get_model()
    
    storage_client = storage.Client()
    
    vector_store = MatchingEngine(
        project_id=PROJECT_ID,
        index=matching_engine_index,
        endpoint=matching_engine_endpoint,
        embedding=EMBEDDING_MODEL,
        gcs_client=storage_client,
        gcs_bucket_name=GCS_BUCKET_NAME
    )
    
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k":5} # number of nearest neighbors to retrieve  
    )
    
    qa = RetrievalQA.from_chain_type(
        llm=generation_model,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True)

    def query(prompt: str):
        response = qa({"query": prompt})
        print(response['result'])
        print("\n\nCitations:")
        for source in response['source_documents']:
            print(source)
            
    response = query(prompt)

    return response
