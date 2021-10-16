import json
import os
import argparse
from tqdm import tqdm
import numpy as np
import random
from multiprocessing import Pool as ProcessPool

MAX_ENTITY_SIZE = 50
MAX_COL_SIZE = 20
MAX_TUPLE_SIZE = 123
MAX_GRAPH_SIZE = 150

from table2txt.graph_strategy.complete_graph import CompleteGraph

def read_table_file(table_lst, data_file, table_filter_set):
    with open(data_file) as f:
        for line in tqdm(f):
            table = json.loads(line)
            table_id = table['tableId']
            if table_filter_set is not None:
                if table_id not in table_filter_set:
                    continue
            table_lst.append(table)
    return table_lst

def read_table_filter_set(table_fileter_file):
    table_id_lst = []
    with open(table_fileter_file) as f:
        for line in f:
            table_id = line.strip()
            table_id_lst.append(table_id)
    table_id_set = set(table_id_lst)
    return table_id_set

def read_tables(table_file, table_filter):
    table_lst = []
    table_filter_set = None
    if table_filter is not None:
        table_filter_set = read_table_filter_set(table_filter)
    read_table_file(table_lst, table_file, table_filter_set)
    return table_lst

def get_topic_entity(table):
    topic_entity = table['documentTitle']
    return topic_entity

def process_table(table):
    strategy = CompleteGraph()
    return strategy.generate(table)

def process_table_2(table):
    topic_entity = get_topic_entity(table)
    columns = table['columns']
    col_name_lst = []
    for col_info in columns:
        col_name = col_info['text']
        col_tokens = col_info['text'].split()
        if len(col_tokens) > MAX_COL_SIZE:
            col_name = ' '.join(col_tokens[:MAX_COL_SIZE])
        col_name_lst.append(col_name)

    row_data = table['rows']
    table_graph_lst = []
    for row_item in row_data:
        cell_lst = row_item['cells']
        row_info = []
        for col_idx, cell in enumerate(cell_lst):
            col_name = col_name_lst[col_idx] 
            cell_text = cell['text']
            cell_token_lst = cell_text.split()
            if len(cell_token_lst) > MAX_ENTITY_SIZE:
                cell_text = ' '.join(cell_token_lst[:MAX_ENTITY_SIZE])
            cell_info = {
                'name':col_name,
                'value':cell_text
            }
            row_info.append(cell_info)
        graph_lst = gen_graph_with_topic_entity(row_info, table['documentUrl'], topic_entity)
        table_graph_lst.extend(graph_lst)
    return (table, table_graph_lst)
    
def main():
    args = get_args()
    table2txt_dir = '/home/cc/code/open_table_discovery/table2txt'
    out_dir = os.path.join(table2txt_dir, 'dataset', args.dataset, args.experiment)
    if os.path.isdir(out_dir):
        err_msg = ('[%s] already exists, please use a different value for [--out_dir].\n'
              % (out_dir))
        print(err_msg)
        return

    os.makedirs(out_dir)

    out_file_src = os.path.join(out_dir, 'test_unseen.source')
    out_file_tar = os.path.join(out_dir, 'test_unseen.target')
    f_o_src = open(out_file_src, 'w')
    f_o_tar = open(out_file_tar, 'w')
    out_row_table_file = os.path.join(out_dir, 'graph_row_table.txt')
    f_o_table = open(out_row_table_file, 'w')

    input_tables = os.path.join('/home/cc/data', args.dataset, 'tables', 'tables.jsonl')
    table_filter_file = os.path.join('/home/cc/data', args.dataset, 'tables', args.table_filter)
    table_lst = read_tables(input_tables, table_filter_file)

    for table in tqdm(table_lst):
        _, graph_lst = process_table(table)
        table_id = table['tableId']
        for graph in graph_lst:
            f_o_src.write(graph + '\n')
            f_o_tar.write('a\n')
            f_o_table.write(table_id + '\n')
        
    f_o_src.close()
    f_o_tar.close()
    f_o_table.close()  
    
def gen_graph_with_topic_entity(row_info, url, topic_entity):
    N = len(row_info)
    tuple_dict = {}
    tuple_info_lst = []

    e_s = topic_entity
    for idx_2 in range(0, N):
        rel = row_info[idx_2]['name'].strip()
        if rel == '':
            continue
        e_o = row_info[idx_2]['value'].strip()
        if e_o == '':
            continue
        tuple_text = '<H> %s <R> %s <T> %s ' % (e_s, rel, e_o)
        tuple_code = tuple_text.lower()
        if tuple_code not in tuple_dict:
            tuple_dict[tuple_code] = 1
            
            token_lst = tuple_text.split()
            tuple_info = {
                'text':tuple_text,
                'token_len':len(token_lst),
                'url':url
            }
            tuple_info_lst.append(tuple_info)
    
    table_tuple_lst = get_table_tuples(row_info, url)
    all_tuple_lst = tuple_info_lst + table_tuple_lst

    graph_lst = tuple2graph(all_tuple_lst)
    return graph_lst

def get_table_tuples(row_info, url):
    N = len(row_info)
    tuple_dict = {}
    tuple_info_lst = []
    for idx_1 in range(0, N):
        e_s_class = row_info[idx_1]['name'].strip()
        e_s = row_info[idx_1]['value'].strip()
        if e_s == '':
            continue
        for idx_2 in range(idx_1+1, N):
            rel = row_info[idx_2]['name'].strip()
            if rel == '':
                rel = ','
            e_o = row_info[idx_2]['value'].strip()
            if e_o == '':
                continue
            tuple_text = '<H> %s <R> %s <T> %s ' % (e_s_class + ' ' + e_s, rel, e_o)
            tuple_code = tuple_text.lower()
            if tuple_code not in tuple_dict:
                tuple_dict[tuple_code] = 1
                
                token_lst = tuple_text.split()
                tuple_info = {
                    'text':tuple_text,
                    'token_len':len(token_lst),
                    'url':url
                }
                tuple_info_lst.append(tuple_info)

    return tuple_info_lst
    #graph_lst = tuple2graph(tuple_info_lst)
    #return graph_lst

def tuple2graph(tuple_info_lst):
    graph_lst = []
    buffer_graphs = []
    buffer_size = 0
    N = len(tuple_info_lst)
    stride = 2
    pos_1 = 0
    while (pos_1 < N):
        pos_2 = pos_1
        while (pos_2 < N):
            tuple_info = tuple_info_lst[pos_2]
            tuple_size = tuple_info['token_len']
            if buffer_size + tuple_size <= MAX_GRAPH_SIZE:
                buffer_graphs.append(tuple_info['text'])
                buffer_size += tuple_size
                pos_2 += 1
            else:
                break

        graph = ''.join(buffer_graphs)
        graph_lst.append(graph)
        buffer_graphs = []
        buffer_size = 0
        if pos_2 >= N:
            break

        next_pos = min(pos_1 + stride, pos_2 - 1)
        if next_pos > pos_1:
            pos_1 = next_pos
        else:
            pos_1 += 1

    return graph_lst

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--table_filter', type=str)
    parser.add_argument('--experiment', type=str)
    parser.add_argument('--debug', type=int, default=0)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()


