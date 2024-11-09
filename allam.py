#!/usr/bin/env python
# coding: utf-8

# ## install

# ## important library

# In[1]:


from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import getpass
from langchain_core.vectorstores import InMemoryVectorStore
import shutil
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from ibm_watsonx_ai.foundation_models import Model
from langchain_ibm import WatsonxLLM
from ibm_watsonx_ai.foundation_models import ModelInference
from langchain_core.prompts import PromptTemplate
from langchain import FewShotPromptTemplate
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader


# ## load data 

# In[2]:


def load_text_with_langchain(file_path):
    # Load the text file
    loader = TextLoader(file_path, encoding='utf-8')
    documents = loader.load()
    
    return documents


file_path = 'D:/Allam/Website/final.txt'
text_documents = load_text_with_langchain(file_path)


# ## split document

# In[3]:


text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=50)
documents=text_splitter.split_documents(text_documents)


# In[6]:


# print(documents[0].page_content)


# ## Embedding

# In[ ]:





# In[4]:


embedding_model = HuggingFaceEmbeddings(
    model_name="Omartificial-Intelligence-Space/mpnet-base-all-nli-triplet-Arabic-mpnet_base",
    model_kwargs={"use_auth_token": "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"}
)


# ## vector database

# In[7]:


vector_store = Chroma(
    collection_name="pdf_documents",
    embedding_function=embedding_model,
    persist_directory="D:/Allam/Website/chroma_db", # Path where the ChromaDB is persisted
)


# In[ ]:


# Generate embeddings

#vector_store.add_documents(documents)


# 

# In[9]:


def get_retrieved_context(question):
    # Fetch relevant context using the retrieval system
    article = vector_store.similarity_search(question,2)
    return article[0].page_content


# ## prompt 

# In[10]:


def format_conversation_history(history):
    """Format conversation history into a string suitable for the prompt."""
    history_text = ""
    for turn in history:
        history_text += f"User: {turn['question']}\nAI: {turn['response']}\n"
    return history_text


# In[11]:


conversation_history = []


# In[ ]:


def generate_grammar_explanation(question,model, history):
    """Generate a grammar explanation with context and conversation history."""
    # Step 1: Retrieve relevant context based on the latest question
    retrieved_context = get_retrieved_context(question)
    
    # Step 2: Format the conversation history for the prompt
    formatted_history = format_conversation_history(history)
    
    # Step 3: Define the prompt using both retrieved context and conversation history
    prompt = f"""[INST]
    المهمة: أنت خبير في قواعد اللغة العربية.

    بناءً على القطعة التالية التي تم استرجاعها من النظام كمصدر معرفي:
    "{retrieved_context}"

    تاريخ المحادثة السابق:
    {formatted_history}

    التوجيه: قم بشرح القواعد النحوية التالية بشكل مفصل وواضح:
    User: {question}
    AI: [/INST]
    """
    
    # # Step 4: Use the model to generate a response
    response = model.generate_text(prompt=prompt, guardrails=False)
    # # Step 5: Update conversation history with the latest question and response
    history.append({"question": question, "response": response})  # Adjust based on model response format

    return response






# In[ ]:





# ## Model

# In[ ]:





# In[13]:


def get_credentials():
    return {
        "url" : "https://eu-de.ml.cloud.ibm.com",
        "apikey" : "Jq8P15FmG-lxwhWU0Zm5mGLVkREuDH4mqTCvy6_UHTg1"
    }


# In[14]:


model_id = "sdaia/allam-1-13b-instruct"
parameters = {
    "decoding_method": "greedy",
    "max_new_tokens": 900,
    "repetition_penalty": 1
}
project_id = "9c0b7793-7781-4139-8205-44ba471b2f82"
space_id = os.getenv("SPACE_ID")


# In[15]:


model = Model(
    model_id = model_id,
    params = parameters,
    credentials = get_credentials(),
    project_id = project_id,
    space_id = space_id
    )


# In[ ]:





# In[ ]:


def rag(question1):
    response1 = generate_grammar_explanation(question1,model, conversation_history)
    return response1


# In[ ]:





# In[ ]:





# ## Delete database

# In[ ]:


# Define the persistence directory where Chroma database is stored
# persist_directory = "/kaggle/working/chroma_db1"  # Replace with your actual directory

# # Check if the directory exists
# if os.path.exists(persist_directory):
#     # Remove the entire directory and all its contents
#     shutil.rmtree(persist_directory)
#     print("The entire Chroma database has been deleted!")
# else:
#     print("The specified persistence directory does not exist.")

