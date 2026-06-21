import streamlit as st
import subprocess
import sys

from rag_with_chunking_level6 import *
sys.path.append('C:\\Users\\sudhir.jain\\')
def query_jupyter(query):
  #Replace with appropriate command to run your Python script
  result = get_result(query)
  
  return result

st.title("RAG ChatBot")
user_query = st.text_input("Enter your query below ")

if user_query:
  response = query_jupyter(user_query)
  st.write(response)