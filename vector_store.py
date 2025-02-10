# vector_store.py
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader
import os

def load_and_split_text(file_path):
    loader = TextLoader(file_path, encoding='utf-8')
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    return text_splitter.split_documents(documents)

embedding_model = HuggingFaceEmbeddings(
    model_name="Omartificial-Intelligence-Space/mpnet-base-all-nli-triplet-Arabic-mpnet_base",
    model_kwargs={"use_auth_token": "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"}

)

vector_store = Chroma(
    collection_name="pdf_documents", embedding_function=embedding_model,
    persist_directory="D:/Allam/Website/chroma_db"
)

def get_retrieved_context(question):
    article = vector_store.similarity_search(question, 2)
    return article[0].page_content if article else ""


def format_conversation_history(history):
    """Format conversation history into a string suitable for the prompt."""
    history_text = ""
    for turn in history:
        history_text += f"User: {turn['question']}\nAI: {turn['response']}\n"
    return history_text

conversation_history = []

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