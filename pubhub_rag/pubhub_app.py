import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import FileSystemBlobLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import PyMuPDFParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from typing_extensions import List, TypedDict

# Language classifier 
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
langcls_model_name = 'qanastek/51-languages-classifier'
tokenizer = AutoTokenizer.from_pretrained(langcls_model_name)
langcls_model = AutoModelForSequenceClassification.from_pretrained(langcls_model_name)
langcls = TextClassificationPipeline(model=langcls_model, tokenizer=tokenizer)

st.title("pub.hub")

#LangChain code
llm = ChatOllama(model="llama3.1:8b")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_store = InMemoryVectorStore(embeddings)

#Load documents
loader = GenericLoader(
    blob_loader=FileSystemBlobLoader(
        path="./pubhub_docs/",
        glob="*.pdf",
    ),
    # blob_parser=PyMuPDFParser(mode="single"),
    blob_parser=PyMuPDFParser(mode="single"),
)
docs = loader.load()

# Text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)
all_splits = text_splitter.split_documents(docs)
document_ids = vector_store.add_documents(documents=all_splits)

# Prompt template
template = """You are an helpful chatbot to support workers at Public Value Technologies gmbh (also known as pub.tech). 
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Always say "thanks for asking!" at the end of the answer.

Respond in {language}.
{context}

Question: {question}

Helpful Answer:"""
prompt = PromptTemplate.from_template(template)

# RAG wit citations

from pydantic import BaseModel, Field

class CitedAnswer(BaseModel):
    """Answer the user question based only on the given sources, and cite the sources used."""

    answer: str = Field(
        ...,
        description="The answer to the user question, which is based only on the given sources.",
    )
    citations: List[int] = Field(
        ...,
        description="The integer IDs of the SPECIFIC sources which justify the answer.",
    )

def format_docs_with_id(docs: List[Document]) -> str:
    formatted = [
        f"Source ID: {i}\nArticle Title: {doc.metadata['source']}\nArticle Snippet: {doc.page_content}"
        for i, doc in enumerate(docs)
    ]
    return "\n\n" + "\n\n".join(formatted)


class State(TypedDict):
    question: str
    language: str
    context: List[Document]
    # answer: str
    answer: CitedAnswer

# Nodes
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"],k=3)
    return {"context": retrieved_docs}


# def generate(state: State):    
#     docs_content = "\n\n".join(doc.page_content for doc in state["context"])
#     messages = prompt.invoke({"question": state["question"], "context": docs_content, "language": state["language"]})
#     # print(messages)
#     response = llm.invoke(messages)
#     return {"answer": response}

def generate(state: State):
    formatted_docs = format_docs_with_id(state["context"])
    messages = prompt.invoke({"question": state["question"], "context": formatted_docs, "language": state["language"]})
    structured_llm = llm.with_structured_output(CitedAnswer)
    response = structured_llm.invoke(messages)
    return {"answer": response}

# Control flow of the program (langgrap)h
from langgraph.graph import START, StateGraph

graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

with st.form("my_form"):
    text = st.text_area(
        "Enter text:",
        "Is workation a possibility?",
    )
    submitted = st.form_submit_button("Submit")
    if submitted: 

        text_lang = langcls(text)[0]

        if text_lang["label"] == 'de-DE':
            response_lang = "German"            
        elif text_lang["label"] == 'it-IT':
            response_lang = "Respond in Italian"            
        else:
            response_lang = "Respond in English"            




        result = graph.invoke({"question": text, "language": response_lang})
        citations_lst = result['answer'].citations
        if len(citations_lst):
            cit_str = ' '.join(list(set([result['context'][cit].metadata['source'] for cit in citations_lst])))
        else:
            cit_str = "No docs to cite."
        st.markdown(result["answer"].answer + f"\n\nReferenced documents: {cit_str} ")