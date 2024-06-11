import json
from promptflow import tool
import chromadb

@tool
def search_in_embeddingDB(queryStr: str, similarity:str) -> tuple[str,str]:
    # 初始化 ChromaDB 客户端
    setting = chromadb.config.Settings(anonymized_telemetry=False)
    chroma_client = chromadb.HttpClient(host='10.1.61.1', port=28000, settings=setting)
    # 假设你的 ChromaDB 中有一个集合，名称为'documents'，并且集合中已经存有文档向量化片段
    #collection = client.get_collection('nnx-mp-curricula-dev_vector')
    collection = chroma_client.get_or_create_collection(name="curricula_help_L2")

    # 从ChromaDB集合中查询相似度最高的3个片段，使用余弦相似度
    results = collection.query(query_texts=queryStr ,n_results=3)
    top_documents=[]
    top_metadatas=[]
    indx=0
    filter_condition=0.5
    # 提取片段内容
    if similarity == "L2":
        filter_condition=1
    else:
        filter_condition=0.5

    for item in results["distances"][0]:
        if item < filter_condition:
            # Perform desired action
            top_documents.append(results['documents'][0][indx])
            top_metadatas.append(results['metadatas'][0][indx])
            indx+=1
    return top_documents,top_metadatas
