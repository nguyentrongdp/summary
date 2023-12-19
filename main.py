from dotenv import load_dotenv
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader, PyPDFLoader
from llm.OpenRouterLLM import OpenRouterLLM
# from langchain.document_loaders import PyPDFLoader

load_dotenv()

# loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
loader = PyPDFLoader("https://arxiv.org/pdf/2106.09680.pdf")
docs = loader.load()

# llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-1106")
llm = OpenRouterLLM(n=1, model='gryphe/mythomist-7b')
chain = load_summarize_chain(llm, chain_type="stuff")

result = chain.run(docs)

print(result)
