# https://note.com/npaka/n/nb9b70619939a

import os
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import CharacterTextSplitter

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_APIKEY') # 'OPENAI_API_KEY'という名前の変数にキーを入れればOK

llm = OpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.2,
    max_tokens=1024,
    # openai_api_key = os.getenv('OPENAI_APIKEY')
    )


# main
def main():
    # テキストの読み込みと分割
    with open("mhi_ceo.txt", encoding="utf-8") as f:
        input_txt = f.read()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    texts = text_splitter.split_text(input_txt)

    # 関連するチャンクの抽出
    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.from_texts(texts, embeddings)
    query = "12代三菱重工の社長は誰ですか"
    docs = docsearch.similarity_search(query)

    chain = load_qa_chain(llm, chain_type="stuff")
    result = chain({"input_documents": docs, "question": query}, return_only_outputs=True)
    # result = chain.run(input_documents=docs, question=query)
    print(result['output_text'])


if __name__ == "__main__":
    main()