import os
import argparse
import datetime
import glob
from trainer import read_config, read_tables, retr_triples, get_train_date_dir
import shutil
import finetune_table_retr as model_tester

def main():
    args = get_args()
    config = read_config()
    table_dict = read_tables(args.work_dir, args.dataset)
    test_query_dir = os.path.join(args.work_dir, 'data', args.dataset, 'query/test')
    
    retr_test_dir = os.path.join(test_query_dir, 'rel_graph')
    if os.path.isdir(retr_test_dir):
        shutil.rmtree(retr_test_dir)

    retr_triples('test', args.work_dir, args.dataset, test_query_dir, table_dict, False, config)
    
    test_args = get_test_args(args.work_dir, args.dataset, retr_test_dir, config)
    model_tester.main(test_args)


def get_date_dir():
    a = datetime.datetime.now()
    test_dir = 'test_%d_%d_%d_%d_%d_%d_%d' % (a.year, a.month, a.day, a.hour, a.minute, a.second, a.microsecond)
    return test_dir


def get_model_file(file_pattern):
        file_lst = glob.glob(file_pattern)
        if len(file_lst) == 0:
            err_msg = 'There is no model file in (%s)' % file_pattern
            raise ValueError(err_msg)
        file_lst.sort(key=os.path.getmtime)
        recent_file = file_lst[-1]
        print('loading recent model file (%s)' % recent_file) 
        return recent_file

def get_test_args(work_dir, dataset, retr_test_dir, config):
    file_name = 'fusion_retrieved_tagged.jsonl'
    eval_file = os.path.join(retr_test_dir, file_name)
    checkpoint_dir = os.path.join(work_dir, 'open_table_discovery/output', dataset)
    checkpoint_name = get_date_dir()
 
    ret_model_file_pattern = os.path.join(work_dir, 'models', dataset, '*.pt') 
    retr_model = get_model_file(ret_model_file_pattern) 
    test_args = argparse.Namespace(sql_batch_no=None,
                                    do_train=False,
                                    model_path=os.path.join(work_dir, 'models/tqa_reader_base'),
                                    fusion_retr_model=retr_model,
                                    eval_data=eval_file,
                                    n_context=int(config['retr_top_n']),
                                    per_gpu_batch_size=int(config['train_batch_size']),
                                    cuda=0,
                                    name=checkpoint_name,
                                    checkpoint_dir=checkpoint_dir,
                                    text_maxlength=int(config['text_maxlength']),
                                    ) 
    return test_args 


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()
