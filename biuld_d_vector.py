import os
import json
import re
import string
from collections import defaultdict, OrderedDict
import math

# 正则表达式用于匹配只包含标点符号的词
punctuation_pattern = re.compile(r'^[{}]+$'.format(re.escape(string.punctuation)))

def is_punctuation(word):
    return punctuation_pattern.match(word) is not None

def build_term_doc_dict(directory, num_files):
    term_doc_dict = defaultdict(lambda: {"doc_freq": 0, "doc_ids": [], "term_freq": defaultdict(int)})

    for num in range(1, num_files + 1):
        if num % 100 == 0:
            print(f"more 100 has down,now {num}")
        filepath = os.path.join(directory, str(num))
        if not os.path.isfile(filepath):
            print(f"文件不存在: {filepath}")
            continue

        try:
            with open(filepath, 'r', encoding='gbk') as f:
                content = f.read()
                words = content.strip().split()
                seen_terms = set()
                for word in words:
                    if is_punctuation(word):
                        continue
                    term_doc_dict[word]["term_freq"][num] += 1#num是当前文档的编号,这部分代码统计了每个词在当前文档中的出现次数
                    if word not in seen_terms:
                        term_doc_dict[word]["doc_ids"].append(num)
                        term_doc_dict[word]["doc_freq"] += 1
                        seen_terms.add(word)
        except Exception as e:
            print(f"读取文件时出错: {filepath}, 错误: {e}")

    return term_doc_dict

def calculate_idf(term_doc_dict, num_files):
    idf_dict = {}
    for term, info in term_doc_dict.items():
        idf_dict[term] = math.log(num_files / (1 + info["doc_freq"]))#t
    return idf_dict

def calculate_tfidf(term_doc_dict, idf_dict, num_files):
    tfidf_dict = defaultdict(dict)
    for term, info in term_doc_dict.items():
        for doc_id in info["doc_ids"]:
            tf = info["term_freq"][doc_id]#n
            tfidf_dict[doc_id][term] = tf * idf_dict[term]
    return tfidf_dict

def save_term_doc_dict(term_doc_dict, output_file):
    sorted_term_doc_dict = OrderedDict(sorted(term_doc_dict.items()))

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sorted_term_doc_dict, f, ensure_ascii=False, indent=4)

def save_tfidf_dict(tfidf_dict, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(tfidf_dict, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    directory = '/Users/wangaiyuan/PycharmProjects/search/data'  # 替换为你的文件目录路径
    num_files = 44970  # 文件数量
    term_doc_output_file = 'vector_dict.json'#在倒排索引表的基础上加上tf df
    tfidf_output_file = 'tfidf.json'

    term_doc_dict = build_term_doc_dict(directory, num_files)
    save_term_doc_dict(term_doc_dict, term_doc_output_file)
    print(f"词典表已保存到 {term_doc_output_file}")

    idf_dict = calculate_idf(term_doc_dict, num_files)
    tfidf_dict = calculate_tfidf(term_doc_dict, idf_dict, num_files)
    save_tfidf_dict(tfidf_dict, tfidf_output_file)
    print(f"TF-IDF权重向量已保存到 {tfidf_output_file}")
