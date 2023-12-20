import streamlit as st
import os
from langchain.chains import RetrievalQA
from langchain.embeddings import VertexAIEmbeddings
from langchain.llms import VertexAI 
from langchain.vectorstores import MatchingEngine
from google.cloud import aiplatform
from google.cloud import storage
import re

PROJECT_ID = 'qwiklabs-asl-04-e7ad74c2e059' #Your Google Cloud Project ID
LOCATION = 'us-central1'   #Your Google Cloud Project Region
EMBEDDING_MODEL = VertexAIEmbeddings(model_name="textembedding-gecko@003")
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
    generation_model = VertexAI(model_name="code-bison-32k", max_output_tokens=8192, temperature=.2)
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

    def get_sources(response):
        source_documents = response.get('source_documents')
        if not source_documents:
            return None
        sources_list = []
        for doc in source_documents:
            doc_parsed = re.search(r'DDL NAME:\s+(.*)\n', doc.page_content).group(1)
            doc_type = re.search(r'DDL_TYPE:\s+(.*)\n', doc.page_content).group(1)
            if doc_type == 'PROCEDURE':
                url = f'https://console.cloud.google.com/bigquery?project={PROJECT_ID}&p={PROJECT_ID}&d=rax_bq_metadata&r={doc_parsed}&page=routine'
            else: url = f'https://console.cloud.google.com/bigquery?project={PROJECT_ID}&p={PROJECT_ID}&d=rax_bq_metadata&t={doc_parsed}&page=table'
            sources_list.append([doc_parsed, url])
        return sources_list


    response = qa({"query": prompt})
    sources = get_sources(response)

    return response, sources
#commenting to force update to test build trigger
