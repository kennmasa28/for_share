import os
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.document_loaders import BSHTMLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import LLMChain
from langchain import PromptTemplate
from bs4 import BeautifulSoup
import requests
import input


def main():
    os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_APIKEY')

    # define llm and chain
    llm = OpenAI(
        model_name="gpt-3.5-turbo-16k",
        temperature=0.2,
        max_tokens=8192,
    )

    chain = load_qa_chain(llm, chain_type="stuff")

    input_info = input.Input()

    # run
    result = chain({"input_documents": input_info.query_related_component, "question": input_info.query},
                   return_only_outputs=True)
    print(result['output_text'])


if __name__ == "__main__":
    main()
