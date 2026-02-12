
import uuid

import google.genai as genai
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
#from IPython.display import Markdown
import os
import getpass
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import WebBaseLoader
from fastapi import FastAPI, HTTPException
#from langgraph.store.memory import InMemoryStore
#from langgraph.checkpoint.memory import MemorySaver, InMemorySaver
from langchain_core.runnables import RunnableConfig
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.store.memory import InMemoryStore
from uuid import uuid4
#import InMemoryStore from "@langchain/core/stores"
from langchain_core.documents import Document
from typing import List, Any
import bs4
from pydantic import BaseModel


class WebContentLoader:
    def __init__(self, urls: List[str]):
        self.urls = urls

    def load_content(self) -> List[Document]:
        loader = WebBaseLoader(self.urls)


        try:
            documents = loader.load()
            print(f"Successfully loaded content from {len(self.urls)} URLs")
            my_list=[]
            for doc in enumerate(documents):
                my_list.append(doc)
                print(my_list)
            return my_list
        except Exception as e:
            print(f"Error loading content: {str(e)}")
            return []


class DocumentChunker:
    def __init__(self, chunk_size: int = 256, chunk_overlap: int = 50):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def create_chunks(self, documents: List[Document]) -> List[Document]:
        chunklist=[]
        for d in documents:
         print("************data*********",d)
         chunks = self.splitter.split_text(d)
         chunklist.append(chunks)
        print(f"Created {len(chunklist)} chunks from {len(documents)} documents")
        print("***********chunks**********",chunklist)
        return chunklist



# Using two of our amazing articles as examples
urls = [
        "https://www.geeksforgeeks.org/nlp/stock-price-prediction-project-using-tensorflow/",
        "https://www.geeksforgeeks.org/deep-learning/training-of-recurrent-neural-networks-rnn-in-tensorflow/"
    ]
webContentLoader =WebContentLoader(urls)

Webdocslist=webContentLoader.load_content()

#pdf
from langchain_community.document_loaders import PyPDFLoader

file_path = './com/xyz/agent/src/samplepdf.pdf'
loader = PyPDFLoader(file_path)
pdfdocslist = loader.load()
print("pdf content",pdfdocslist[0].page_content)

if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

from google import genai

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

from google.genai import types

client = genai.Client()
for m in client.models.list():
  if 'embedContent' in m.supported_actions:
    print('avialable models:',m.name)

from google.genai import types


class GeminiEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        pass
    def __call__(self, input: Documents) -> Embeddings:
        EMBEDDING_MODEL_ID = "models/gemini-embedding-001"  # @param ["gemini-embedding-001", "text-embedding-004"] {"allow-input": true, "isTemplate": true}
        title = "Custom query"
        response = client.models.embed_content(
            model=EMBEDDING_MODEL_ID,
            contents=input,
            config=types.EmbedContentConfig(
                task_type="retrieval_document",
                title=title
            )
        )
        print("embeddings:",response.embeddings[0].values[:5])
        return response.embeddings[0].values



DOCUMENT1 =Document(metadata={'source': 'https://www.geeksforgeeks.org/', 'title': 'Training', 'description': 'desc.', 'language': 'en'},page_content='Ramkiran is a AI developer and from INDIA.He is 45 years and from Hyderabad.')
DOCUMENT2 = Document(metadata={'source': 'https://www.geeksforgeeks.org/', 'title': 'Training', 'description': 'desc.', 'language': 'en'},page_content='Shifting Gears Your Googlecar has an automatic transmission. Toshift gears, simply move the shift lever to the desired position.')

DOCUMENT3 = Document(metadata={'source': 'https://www.geeksforgeeks.org/', 'title': 'Training', 'description': 'desc.', 'language': 'en'},page_content='LangChain provides a framework for building applications powered by large language models (LLMs). The workflow in LangChain can be understood in terms of its core components and how they interact to achieve a desired outcome.')


documents = [DOCUMENT1.page_content, DOCUMENT2.page_content,DOCUMENT3.page_content]


for d in Webdocslist:
    documents.append(d)
for d in pdfdocslist[0].page_content:
    documents.append(d)

print('Final documents list:',documents)

documentChunker = DocumentChunker()
#chunklist=documentChunker.create_chunks(documents)
#print("web based loader chunklist:",chunklist)


# 2. Setup ChromaDB
# For a persistent database (saved to disk), use output_path. For RAM-only, remove the path.
def create_chroma_db(documents, name):
  chroma_client = chromadb.Client()
  db = chroma_client.create_collection(
      name=name,
      embedding_function=GeminiEmbeddingFunction()
  )

  for i, d in enumerate(documents):
    db.add(
      documents=d,
      ids=str(i)
    )
  return db

# Set up the DB
db = create_chroma_db(documents, "google-car-db2")
def call_chromaDB(query, db):
    print("query in chromadb:",query)
    response = db.query(query_texts=[query], n_results=1)['documents'][0][0]
    print(" chroma DB response",response)
    return response




from langchain.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
import operator


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int=0
    query:str
    context:str
    thread_id: str
    user_id: str

# Step 3: Define model node
from langchain.messages import SystemMessage

#from langgraph.nodes import LLMNode
from langgraph.prebuilt import ToolNode
#from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain.tools import tool
from langchain.chat_models import init_chat_model

#@tool("retrivernode")
def retriever_node(state: dict) -> dict[str, Any]:
    # Perform embedding search
 query=state["messages"][-1].content
 #response = call_chromaDB(state['query'], db)
 response = call_chromaDB(query, db)
 #Markdown(passage)
 #set context
 context=response
 print("retriver node response for setting context:", context)
 SystemMessage = (
        "You are a helpful assistant. Use the following context in your response:"
        f"\n\n{context}"
 )
 return {"messages":[SystemMessage],"context":context,"query":query}
# 2. Define nodes

#tools = [retriever_node]
#tool_node = ToolNode(tools)
from langchain.messages import HumanMessage


#llm = init_chat_model("google_genai:gemini-2.5-flash-lite")
llm = init_chat_model("google_genai:gemini-3-flash-preview")
# Augment the LLM with tools
#llm_with_tools = llm.bind_tools([retriever_node])
#llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
def llm_call(state: dict):
    #query=state["messages"][-1].content
    """LLM decides whether to call a tool or not"""
    print('llm call with state query:',state['query'])
    print('llm call with state context:', state['context'])
    print('llm call with state messages list:', state['messages'])
    return {
        "messages": [
            llm.invoke(
                [HumanMessage(content=state['query']),
                    SystemMessage(
                        content="You are a helpful assistant. Use the following context in your response:{context}"
                    )
                ]
                + state["messages"]
            )
        ],
        "llm_calls": state.get('llm_calls', 0) + 1
    }
# 3. Build graph
#graph = Graph()
workflow  = StateGraph(MessagesState)
workflow .add_node("llm_call", llm_call)
workflow .add_node("tool_node", retriever_node)

# Connect retriever → generator

# Add edges to connect nodes
workflow .add_edge(START, "tool_node")#retriver
workflow .add_edge("tool_node", "llm_call")#generator
workflow .add_edge("llm_call",END)
#checkpointer = MemorySaver()  # Use PostgresSaver for production
#checkpointer = InMemorySaver()
checkpointer = MemorySaver()  # Use PostgresSaver for production

#memory = MemorySaver()
#in_memory_store = InMemoryStore()
#memory_store = MemoryStore()
#graph = workflow.compile(store=in_memory_store)
#graph = workflow.compile(checkpointer=checkpointer,store=in_memory_store)
graph = workflow.compile(checkpointer=checkpointer)
#graph = workflow.compile(checkpointer=memory)

#graph = workflow.compile(checkpointer=checkpointer)

#
# 4. Run workflow
# query = "tell about ramkiran"
# from langchain.messages import HumanMessage
# messages = [HumanMessage(content=query)]
# Prepare config for state persistence
# Define a config with a thread ID.
# thread_id = uuid.uuid4()
# print('thread_id',thread_id)
#config = {"configurable": {"thread_id": thread_id}}
#config: RunnableConfig = {"configurable": {"thread_id": "1"}}
# config = {"configurable": {"thread_id": thread_id, "user_id": "user_id1"}}
# messages = graph.invoke({"messages": messages},config=config)
# for m in messages["messages"]:
#    print(m)


# query = "hi i am ramkiran"
# messages = [HumanMessage(content=query)]
# messages = graph.invoke({"messages": messages},config)
# for m in messages["messages"]:
#    print(m)
#
# query = "what is my name"
# messages = [HumanMessage(content=query)]
# messages = graph.invoke({"messages": messages},config)
# for m in messages["messages"]:
#     print(m)
# query = "what is tensorflow"
# messages = [HumanMessage(content=query)]
# messages = graph.invoke({"messages": messages}, config)
# for m in messages["messages"]:
#     print(m)
# print("current state",graph.get_state(config))
# print("state history",list(graph.get_state_history(config)))
#



# 2. Define Request Schema
class ChatRequest(BaseModel):
        message: str
        thread_id: str = "default_thread"
        user_id:str = "user1"

# 3. Create Endpoint
app = FastAPI()

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # Prepare input state
        query=request.message
        print("query in chat_endpoint",query)
        #input_state = {"messages": [HumanMessage(content=query)]}
        input_message =  {"messages":[HumanMessage(content=query)]}
        # Prepare config for state persistence
        config = {"configurable": {"thread_id": request.thread_id,"user_id": request.user_id}}

        # Invoke Graph
        # Use ainvoke for non-blocking async execution
        #result = await graph.ainvoke(input_message, config=config)
        result = graph.invoke(input_message, config=config,query=query)
        # Extract the last message content
        last_message = result["messages"][-1].content
        return {"response": last_message, "thread_id": request.thread_id,"user_id": request.user_id}

    except Exception as e:
     raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
 import uvicorn


 uvicorn.run(app, host="localhost", port=8000)



