#from vertexai.preview.language_models import TextGenerationModel
import vertexai
import streamlit as st
import os
import langchain.chains.RetrievalQA
import langchain.embeddings.VertexAIEmbeddings
import langchain.llms.VertexAI 
import langchain.vectorstores.MatchingEngine
from google.cloud import aiplatform
from google.cloud import storage

PROJECT_ID = 'qwiklabs-asl-04-e7ad74c2e059' #Your Google Cloud Project ID
LOCATION = 'us-central1'   #Your Google Cloud Project Region
vertexai.init(project=PROJECT_ID, location=LOCATION)
GCS_BUCKET_NAME = f'{PROJECT_ID}-{DISPLAY_NAME}'
EMBEDDING_MODEL = VertexAIEmbeddings(model_name="textembedding-gecko")
ENDPOINT_ID = '7083674030691057664'
INDEX_ID = '5317137076854980608'
GCS_BUCKET_NAME='qwiklabs-asl-04-e7ad74c2e059-data_delver_v2'
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
