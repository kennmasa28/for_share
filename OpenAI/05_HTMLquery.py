import os
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.chains import LLMChain
from langchain import PromptTemplate
from bs4 import BeautifulSoup
import requests

# HTMLファイルの読み込み
url = "https://shibaken.sakura.ne.jp/career.html"
response = requests.get(url)
html_content = response.text
loader = UnstructuredHTMLLoader(html_content)
data = loader.load()

def preprocess_data(data):
    # HTMLをパースしてテキスト部分のみを抽出
    soup = BeautifulSoup(data.page_content, 'html.parser')
    text = soup.get_text()
    return text

# 質問応答のチェーンの作成
llm = OpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.9,
    max_tokens=4000,
    openai_api_key = os.getenv('OPENAI_APIKEY')
    )

prompt = PromptTemplate(
    input_variables=["formatted_data"],
    template="この人はどこの会社で働いていますか。：{formatted_data}",
)

chain = LLMChain(llm=llm, prompt=prompt)

# データの整形（必要に応じて）
formatted_data = preprocess_data(data)

# チェーンの実行
response = chain.run(formatted_data)

# 応答の表示
if response:
    response_text = str(response)  # 必要に応じて、responseを文字列に変換する
    with open("output_html.txt", "w") as file:
        file.write(response_text)