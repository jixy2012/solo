import os
import random
import json
from tqdm import tqdm
import argparse
import uuid
import sys
from typing import Dict, List
import shutil

def set_python_path():

    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Assuming your directory structure is similar and you want to go up one directory from the current script's location
    work_dir = os.path.dirname(current_dir)

    # Construct the paths you want to add to PYTHONPATH
    paths_to_add = [
        os.path.join(work_dir, 'open_table_discovery'),
        os.path.join(work_dir, 'open_table_discovery', 'demo'),
        os.path.join(work_dir, 'open_table_discovery', 'relevance'),
        os.path.join(work_dir, 'open_table_discovery', 'sql2question'),
    ]

    # Prepend these paths to sys.path
    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)
    # for path in sys.path:
    #     print(path)
set_python_path()

import tester

def create_query_file(qry_data_dir, question):
    file_name = 'fusion_query.jsonl'
    data_file = os.path.join(qry_data_dir, file_name)
    query_data = {
        "id":0,
        "question":question,
        "table_id_lst":['N/A'],
        "answers":["N/A"],
        "ctxs": [{"title": "", "text": "This is a example passage."}]
    }
    with open(data_file, 'w') as f_o:
        f_o.write(json.dumps(query_data) + '\n')

def get_top_tables(out_dir):
    data_file = os.path.join(out_dir, 'pred_epoch_0_None.jsonl')
    with open(data_file) as f:
        item = json.load(f)
    out_table_lst = []
    out_table_set = set()
    tag_lst = item['tags']
    for tag in tag_lst:
        table_id = tag['table_id']
        if table_id not in out_table_set:
            out_table_set.add(table_id)
            out_table_lst.append(table_id)
            if len(out_table_lst) >= 5:
                break
    print(out_table_lst)
    return out_table_lst


# returns the table id from running a query
def main(question: str, dataset: str) -> Dict:
    random.seed(0)
    work_dir = os.path.dirname(os.getcwd())
    data_dir = os.path.join(work_dir, 'data')
    args = argparse.Namespace(work_dir=work_dir, dataset=dataset, data_dir=data_dir, table_repre='rel_graph')
    table_dict =load_tables(args)
    index_obj = tester.get_index_obj(args.work_dir, dataset, args)
    print(f"args: {args}")
    output_table_lst = query(question, args, table_dict, index_obj)
    print(f"answer: {output_table_lst}")
    return output_table_lst

def load_tables(args: argparse.Namespace) -> Dict[str, Dict]:
    data_dir = args.data_dir
    dataset = args.dataset
    table_dict = {}
    data_file = os.path.join(data_dir, dataset, 'tables/tables.jsonl')
    with open(data_file) as f:
        for line in tqdm(f):
            table_data = json.loads(line)
            table_id = table_data['tableId']
            table_dict[table_id] = table_data
    return table_dict

def create_test_args(qry_folder: str, args: argparse.Namespace):
    test_args = argparse.Namespace(
        work_dir=args.work_dir,
        dataset=args.dataset,
        query_dir=qry_folder,
        table_repre='rel_graph',
        train_model_dir=None,
        bnn=1,
    )
    return test_args

def query(question: str, args: argparse.Namespace, table_dict: Dict[str, Dict], index_obj) -> List:
    
    data_dir = args.data_dir
    dataset = args.dataset
    qry_folder = 'demo_query_%s' % str(uuid.uuid4())
    query_dir = os.path.join(data_dir, dataset, qry_folder)
    test_args = create_test_args(qry_folder, args)
    print("test_args: ", test_args)
    qry_data_dir = os.path.join(query_dir, 'test')
    if not os.path.isdir(qry_data_dir):
        os.makedirs(qry_data_dir)
    create_query_file(qry_data_dir, question)
    out_dir = tester.main(test_args, table_data=table_dict, index_obj=index_obj)
    top_table_lst = get_top_tables(out_dir)
    table_data_lst = [table_dict[table_id] for table_id in top_table_lst]
    rank = 0
    for table_data in table_data_lst:
        rank += 1
        table_data['search_rank'] = rank
        
    shutil.rmtree(query_dir, ignore_errors=True)
    shutil.rmtree(out_dir, ignore_errors=True)
     
    return table_data_lst


main("how old is Linda Taylor?", "fetaqa")