import sys
def set_python_path():

    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))

    work_dir = os.path.dirname(current_dir)

    # Construct the paths to add to PYTHONPATH
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

import argparse
import os
from tqdm import tqdm
import glob
from table2txt import table2graph
import table_from_csv
import generate_passage_embeddings as passage_encoder
from src import ondisk_index
import shutil
import json
from trainer import read_config
from typing import Dict, List
StateImportCSV = 'import_csv'
StateGenTriples = 'gen_triples'
StateEncode = 'encode'
StateIndex = 'index'
EmbFileTag = '_embeddings'



def get_state_file(dataset: str) -> str:
    '''
    return the file name for the current dataset's state file. state file provides intermediate 
    info about the progress of indexing the data set
    '''
    return f'index_state_{dataset}.json'

def read_state(state_file: str) -> Dict:
    '''
    read in the current state of processing this table/dataset.
    args:
        state_file: path to the state file. new one will be created if this doesn't exist
    states:
        StateImportCSV: whether the csv files have been successfully imported
        StateGenTriples: whether the tables have been processed to generate the tables
        StateEncode: whether the table encodings have been completed
        StateIndex: whether the indices have been created (carl note: diff between encode and index?)
    '''
    if os.path.isfile(state_file):
        with open(state_file) as f:
            state_info = json.load(f)
    else:
        state_info = {
            StateImportCSV: False,
            StateGenTriples: False,
            StateEncode: False,
            StateIndex: False
        }
    return state_info

def update_state(state_info: Dict, state_key: str, state: str, state_file: str) -> None:
    '''
    update the state dictionary & state file
    args:
        state_info: current state's dictionary
        state_key: key in the dictionary
        state: the new state
        state_file: file to save the state dictionary
    '''
    state_info[state_key] = state
    with open(state_file, 'w') as f_o:
        f_o.write(json.dumps(state_info))


def get_csv_args(work_dir, dataset, config):
    csv_args = argparse.Namespace(work_dir=work_dir,
                                  dataset=dataset,
                                  file_name_title=config['file_name_title'],
                                  table_sample_rows=config['table_sample_rows']
                                 )
    return csv_args 

def get_graph_args(work_dir, dataset, config):
    graph_args = argparse.Namespace(work_dir=work_dir, 
                                    dataset=dataset,
                                    experiment='rel_graph',
                                    table_file='tables.jsonl',
                                    strategy='RelationGraph',
                                    table_chunk_size=config['table_chunk_size'],
                                    table_import_batch=config['table_import_batch']
                                    )
    return graph_args

def get_encoder_args(model_path, config, show_progress=True):
    encoder_args = argparse.Namespace(is_student=True,
                                      passages=None, 
                                      output_path=None,
                                      output_batch_size=500000,
                                      shard_id=0,
                                      num_shards=1,
                                      per_gpu_batch_size=config['encode_batch_size'],
                                      passage_maxlength=200,
                                      model_path=model_path,
                                      no_fp16=False,
                                      show_progress=show_progress
                                     )
    return encoder_args

def get_index_args(work_dir, dataset):
    emb_file = get_emb_file_pattern(work_dir, dataset)
    index_args = argparse.Namespace(work_dir=work_dir,
                                    dataset=dataset,
                                    experiment='rel_graph',
                                    emb_file=emb_file
                                    )
    return index_args 

def exists_tables_csv(dataset_dir):
    csv_file_pattern = os.path.join(dataset_dir, 'tables_csv', '**', '*.csv')
    csv_file_lst = glob.glob(csv_file_pattern, recursive=True)
    return len(csv_file_lst) > 0

def get_emb_file_pattern(work_dir, dataset):
    emb_file = os.path.join(work_dir, 'open_table_discovery/table2txt/dataset', 
                            dataset, 'rel_graph', 'emb', '*%s*' % EmbFileTag)
    return emb_file

def confirm(args):
    dataset_dir = os.path.join(args.work_dir, 'data', args.dataset)
    tables_file = os.path.join(dataset_dir, 'tables/tables.jsonl')
    passage_dir = os.path.join(args.work_dir, 'open_table_discovery/table2txt/dataset', 
                                args.dataset, 'rel_graph')
    passage_file = os.path.join(passage_dir, 'passages.jsonl') 
    emb_file = get_emb_file_pattern(args.work_dir, args.dataset) 
    index_dir = os.path.join(args.work_dir, 'index/on_disk_index_%s_rel_graph' % args.dataset)     
 
    check_data_lst = [] 
    if (args.pipe_step is None) or (args.pipe_step == ''): 
        tables_csv_exists = exists_tables_csv(dataset_dir)
        args.tables_csv_exists = tables_csv_exists 
        table_exists = os.path.exists(tables_file)
        passage_exists = os.path.exists(passage_file)
        emb_file_lst = glob.glob(emb_file) 
        emb_exists = len(emb_file_lst) > 0
        
        if tables_csv_exists:
            if table_exists:
                check_data = {'name': 'Tables imported', 'file_lst': [tables_file]}
                check_data_lst.append(check_data) 
        if passage_exists:
            check_data = {'name':'Triples', 'file_lst': [passage_file]}
            check_data_lst.append(check_data) 
        if emb_exists:
            check_data = {'name':'Triple embeddings', 'file_lst':emb_file_lst}
            check_data_lst.append(check_data) 
    
    index_exists = os.path.exists(index_dir)
    if index_exists:
        check_data = {'name':'Index', 'dir':index_dir}
        check_data_lst.append(check_data)
             
    if len(check_data_lst) > 0:
        check_data_desc = get_check_data_desc(check_data_lst) 
        confirmed = input('%s already exists. If continue, the data will be removed and recreated. \n' % check_data_desc +
                          'Do you want to continue(y/n)? ')
        if confirmed.lower().strip() == 'y':
            clear_checked_data(check_data_lst)
            return True
        else:
            return False
    else:
        return True

def clear_checked_data(check_data_lst):
    for check_data in check_data_lst:
        file_lst = check_data.get('file_lst', [])
        for file_path in file_lst:
            os.remove(file_path) 
        data_dir = check_data.get('dir', None)
        if data_dir is not None:
            shutil.rmtree(data_dir)

def get_check_data_desc(check_data_lst):
    desc = '('
    for offset, check_data in enumerate(check_data_lst):
        desc += check_data['name']
        if offset < len(check_data_lst) - 1:
            desc += ' , '
    desc += ')'
    return desc

def index_dataset(dataset: str):
    work_dir = os.path.dirname(os.getcwd())
    data_dir = os.path.join(work_dir, 'data')
    args = argparse.Namespace(work_dir=work_dir, dataset=dataset, data_dir=data_dir, pipe_step=None)
    pipe_state_file = get_state_file(args.dataset)
    pipe_state_info = read_state(pipe_state_file)
    if not confirm(args):
        return
    config = read_config()
    if (args.pipe_step is not None) and (args.pipe_step != ''):
        if args.pipe_step != 'emb_to_index':
            print('arg pipe_step only support "emb_to_index"')
            return
        else:
            pipe_triple_file = './table2txt/dataset/%s/rel_graph/passages.jsonl' % args.dataset
            create_index(pipe_state_info, pipe_state_file, args, pipe_triple_file)
            return
             
    if args.tables_csv_exists:
        import_table_msg = '\nImporting tables'
        if config['table_sample_rows'] is not None:
            import_table_msg += '(Sample rows)'
        print(import_table_msg)
        csv_args = get_csv_args(args.work_dir, args.dataset, config)
        msg_info = table_from_csv.main(csv_args)
        if not msg_info['state']:
            update_state(pipe_state_info, StateImportCSV, False, pipe_state_file)
            print(msg_info['msg'])
            return
        else:
            update_state(pipe_state_info, StateImportCSV, True, pipe_state_file)
    
    print('\nGenerating triples')
    graph_args = get_graph_args(args.work_dir, args.dataset, config)
    msg_info = table2graph.main(graph_args)
    graph_ok = msg_info['state']
    if not graph_ok:
        update_state(pipe_state_info, StateGenTriples, False, pipe_state_file)
        return
    else:
        update_state(pipe_state_info, StateGenTriples, True, pipe_state_file)
    num_triples = msg_info['num_triples'] 
    triple_file = msg_info['out_file']
    print('\nEncoding triples')
    encode_triples(args.work_dir, triple_file, config)
    update_state(pipe_state_info, StateEncode, True, pipe_state_file)
    
    #Creating index  
    create_index(pipe_state_info, pipe_state_file, args, triple_file) 

def create_index(pipe_state_info, pipe_state_file, args, triple_file):
    emb_file_pattern = get_emb_file_pattern(args.work_dir, args.dataset) 
    out_emb_file_lst = glob.glob(emb_file_pattern) 
    
    if len(out_emb_file_lst) == 0:
        raise ValueError('There is no triple embedding files')
    
    print('\nCreating index')
    index_args = get_index_args(args.work_dir, args.dataset)
    msg_info = ondisk_index.main(index_args)
    if pipe_state_info is not None:
        if not msg_info['state']:
            update_state(pipe_state_info, StateIndex, False, pipe_state_file)
            print(msg_info['msg'])
        else:
            update_state(pipe_state_info, StateIndex, True, pipe_state_file)

    index_dir = msg_info['index_dir']
    assert(os.path.isdir(index_dir))
    shutil.move(triple_file, index_dir)

    for out_emb_file in out_emb_file_lst:
        os.remove(out_emb_file)
    
    print('\nIndexing done')
     

def encode_triples(work_dir, graph_file, config):
    print('Encoding %s' % graph_file)
    encoder_model = os.path.join(work_dir, 'models/student_tqa_retriever_step_29500')
    out_emb_file_lst = []
    encoder_args = get_encoder_args(encoder_model, config, show_progress=False)
    encoder_args.passages = graph_file
    passage_dir = os.path.dirname(graph_file)
    base_name = os.path.basename(graph_file)
    emb_dir = os.path.join(passage_dir, 'emb')
    if not os.path.isdir(emb_dir):
        os.makedirs(emb_dir)
    encoder_args.output_path = os.path.join(emb_dir, base_name + EmbFileTag)
    passage_encoder.main(encoder_args, is_main=False) 

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--pipe_step', type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()

