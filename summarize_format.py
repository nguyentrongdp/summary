from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from dotenv import load_dotenv

load_dotenv()

# This controls how each document will be formatted. Specifically,
# it will be passed to `format_document` - see that function for more
# details.
document_prompt = PromptTemplate(
    input_variables=["page_content"],
    template="{page_content}"
)

# Define prompt
prompt_template = """
Given this content below, return a JSON object with the following keys:
- summary: Summary of the document.
- keywords: Top 50 keywords of the document.
- category: the category of the document, category include 'Medicine', 'Chemistry', 'Biology', 'Humanities', 'Physics', 'Engineering', 'Math', 'Ecology', 'Computer Science', 'Economics', 'Geosci'.

Don't narrate, just answer with a JSON object, nothing else.

Document content:
{document_content}
"""

prompt = ChatPromptTemplate.from_template(prompt_template)

# Define LLM chain
llm = ChatOpenAI(temperature=0, model_name="gpt-4-1106-preview",
                 model_kwargs={"response_format": {"type": "json_object"}})

llm_chain = LLMChain(llm=llm, prompt=prompt)

# Define StuffDocumentsChain
stuff_chain = StuffDocumentsChain(
    llm_chain=llm_chain,
    document_prompt=document_prompt,
    document_variable_name="document_content"
)


def summarize_doc(file_path: str):
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    return stuff_chain.run(docs)


if __name__ == '__main__':
    res = summarize_doc('./docs/15440478.2020.1818344.pdf')
    res = eval(res)
    print(
        f'Summary: {res["summary"]}\nKeyword: {res["keywords"]}\nCategory: {res["category"]}')
