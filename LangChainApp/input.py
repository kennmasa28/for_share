import os
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.document_loaders import BSHTMLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import LLMChain
from langchain import PromptTemplate
from bs4 import BeautifulSoup
import requests


class Input():

    def __init__(self):
        self.query = "三菱原子力工業株式会社を合併したのはいつですか"
        self.source_path = "mhi.txt"
        self.full_source_text = self.GetSourceFromText(self.source_path)
        self.query_related_component = self.QueryRelatedComponent(
            self.full_source_text, self.query)

    def GetSourceFromText(self, path):
        with open(path, encoding="utf-8") as f:
            input_txt = f.read()
        return input_txt

    def GetSourceFromCSV(self, path):
        loader = CSVLoader(file_path=path, csv_args={
            'delimiter': ',',
            'quotechar': '"',
            'fieldnames': ['year', 'sell', 'profit']
        })
        data = loader.load()
        return data

    def GetSourceFromHTML(self, url):
        response = requests.get(url)
        html_content = response.text
        soup = BeautifulSoup(html_content, 'html.parser',
                             from_encoding='utf-8')
        text_content = soup.get_text()
        return text_content

    def QueryRelatedComponent(self, full_source_text, query):
        text_splitter = CharacterTextSplitter(chunk_size=5000, chunk_overlap=0)
        texts = text_splitter.split_text(full_source_text)
        embeddings = OpenAIEmbeddings()
        docsearch = FAISS.from_texts(texts, embeddings)
        query_related_component = docsearch.similarity_search(query)
        return query_related_component
