
import json
import re
import string
from collections import defaultdict, OrderedDict
import math
#计算向量相关度
def load_tfidf(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        tfidf_dict = json.load(f)
    return tfidf_dict

def cosine_similarity(vec1, vec2):#计算余弦相似度
    dot_product = sum(vec1[term] * vec2.get(term, 0.0) for term in vec1)
    magnitude_vec1 = math.sqrt(sum(value ** 2 for value in vec1.values()))#模
    magnitude_vec2 = math.sqrt(sum(value ** 2 for value in vec2.values()))
    if magnitude_vec1 == 0 or magnitude_vec2 == 0:
        return 0.0
    return dot_product / (magnitude_vec1 * magnitude_vec2)#点积/模

def rank_documents(query_tfidf, tfidf_dict, top_k=10):
    similarity_scores = []
    for doc_id, doc_tfidf in tfidf_dict.items():
        similarity = cosine_similarity(query_tfidf, doc_tfidf)
        similarity_scores.append((doc_id, similarity))

    # 按相似度从高到低排序
    similarity_scores.sort(key=lambda x: x[1], reverse=True)

    # 返回前K篇文档
    return similarity_scores[:top_k]
#正则表达式用于匹配只包含标点符号的词
punctuation_pattern = re.compile(r'^[{}]+$'.format(re.escape(string.punctuation)))


def is_punctuation(word):
    return punctuation_pattern.match(word) is not None

def process_query(query, idf_dict):#计算查询的权重向量
    # 分词并计算词频（TF）
    query_terms = query.strip().split()
    term_freq = defaultdict(int)
    for term in query_terms:
        if is_punctuation(term):
            continue
        term_freq[term] += 1

    # 计算查询的TF-IDF权重向量
    query_tfidf = {}
    for term, tf in term_freq.items():
        if term in idf_dict:
            tf=0 if tf==0 else math.log(tf)+1#平滑单词频率
            query_tfidf[term] = tf * idf_dict[term]
        else:
            query_tfidf[term] = 0.0  # 如果查询中有在文档中未出现的词，IDF设为0

    return query_tfidf
def save_idf_dict(idf_dict, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(idf_dict, f, ensure_ascii=False, indent=4)
def calculate_idf(term_doc_dict, num_files):
    idf_dict = {}
    for term, info in term_doc_dict.items():
        idf_dict[term] = math.log(num_files / (1 + info["doc_freq"]))
    return idf_dict
#保存一次查询的topk结果
def save_result(top_documents,result_file):
    with open(result_file,'a')as f:
        for doc_id, similarity in top_documents:
            print(f"Document ID: {doc_id}, Similarity: {similarity}",end=',')
            f.write(f"{doc_id}\t")
        print("\n")
        f.write("\n")

num_files = 44970  # 文件数量
# 读取 JSON 文件并将内容加载到 Python 字典中
input_file = 'vector_dict.json'
idf_output_file='didf.json'#用于计算查询的
with open(input_file, 'r', encoding='utf-8') as f:
    term_doc_dict = json.load(f)
#idf_dict = calculate_idf(term_doc_dict, num_files)#每个词的idf值
#save_idf_dict(idf_dict, idf_output_file)
with open(idf_output_file, 'r', encoding='utf-8') as f:
    idf_dict = json.load(f)
print(f"IDF字典已保存到 {idf_output_file}")

test_file='test_list.txt'#测试数据文件
result_file='result1.txt'
tfidf_file_path = 'tfidf.json'
tfidf_dict = load_tfidf(tfidf_file_path)
with open(test_file,'r')as f:
    for line in f:

        query_tfidf=process_query(line,idf_dict)
        top_k = 10
        top_documents = rank_documents(query_tfidf, tfidf_dict, top_k)
        save_result(top_documents,result_file)






