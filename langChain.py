from langchain import hub
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_core.language_models.llms import LLM
from langchain_core.embeddings import Embeddings
from langchain_community.document_loaders import TextLoader
from langchain_core.vectorstores import InMemoryVectorStore

# Load and preprocess documents
loader = TextLoader("docs.txt")  # Replace with your document file
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

class CustomEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        return [[0.5, 0.6, 0.7] for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return self.embed_documents([text])[0]

# Store embeddings
embeddings = CustomEmbeddings()
vector_store = InMemoryVectorStore(embeddings)

# Index chunks
_ = vector_store.add_documents(documents=docs)

# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# Define the model API call
class CustomLLM(LLM):
    def _call(self, prompt, stop=None):
        print(prompt)
        return prompt[: 3]

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model. Used for logging purposes only."""
        return "custom"
    
llm = CustomLLM()

# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    response = llm.invoke(docs_content)
    return {"answer": response}


# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

response = graph.invoke({"question": "What is my mother name?"})
print(response["answer"])