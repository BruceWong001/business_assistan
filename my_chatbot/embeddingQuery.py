import json
from promptflow import tool
import chromadb

@tool
def search_in_embeddingDB(queryStr: str) -> str:
    # 初始化 ChromaDB 客户端
    setting = chromadb.config.Settings(anonymized_telemetry=False)
    chroma_client = chromadb.HttpClient(host='localhost', port=8080, settings=setting)
    # 假设你的 ChromaDB 中有一个集合，名称为'documents'，并且集合中已经存有文档向量化片段
    #collection = client.get_collection('nnx-mp-curricula-dev_vector')
    collection = chroma_client.get_or_create_collection(name="test")

    # 从ChromaDB集合中查询相似度最高的3个片段，使用余弦相似度
    results = collection.query(query_texts=queryStr ,n_results=3)
    # 提取片段内容
    top_documents = results['documents'][0]

    # 将结果转换为JSON数组字符串
    json_result =top_documents # json.dumps(top_documents, ensure_ascii=False)

    return json_result
