import re
import string
from collections import defaultdict
import math

# 正则表达式用于匹配只包含标点符号的词
punctuation_pattern = re.compile(r'^[{}]+$'.format(re.escape(string.punctuation)))

def is_punctuation(word):
    return punctuation_pattern.match(word) is not None

def process_query(query, idf_dict):
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
            query_tfidf[term] = tf * idf_dict[term]
        else:
            query_tfidf[term] = 0.0  # 如果查询中有在文档中未出现的词，IDF设为0

    return query_tfidf

num_files = 44970  # 文件数量# 打开文件并逐行读取
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        # 去掉行末的换行符
        line = line.strip()
        # 分割字符串
        parts = line.split(' ', 1)  # 只分割第一个空格，以防止行中有多个空格
        if len(parts) == 2:
            str1, str2 = parts
            # 处理读取到的字符串
            merged_doc_ids=and_hebing(str1,str2,term_doc_dict)
            #merged_doc_ids=or_hebing(str1,str2,term_doc_dict)

            if merged_doc_ids:
                #保存该次的结果
                save_merged_doc_ids(merged_doc_ids,output_file)
                num+=1
            else:
                print(f"{str1} 和 {str2} 没有共同的文档。")

        else:
            print(f'Line does not contain two strings separated by a space: {line}')

term_doc_dict = load_term_doc_dict(term_doc_file)
num=0

