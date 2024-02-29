import os
import shutil
import uuid
from target_run_query import run_query, set_python_path
from target_index_tables import index_dataset
from typing import List
class SoloCustomModel():

    def __init__(self, model_path: str):
        self.model_path = model_path
        set_python_path()


    def copy_tables_to_data_folder(
            self, 
            tables_path: List[str], 
            dataset_name: str
            ) -> None:
        '''
        solo is set up so that the tables are stored in a special folder for any processing. i'm not gonna break this for now bc most of the code base depends on structuring the datasets in a very particular way. in order to achieve generality, i've decided to keep the interface's encode_tables method to still feed in a bunch of table csv files, and i'll simply copy over those files to a folder in the `data` directory for further processing. just having solo deal with the rest.
        '''
        # Construct the target directory path
        target_dir = os.path.join('../data', dataset_name, 'tables_csv')

        # Create the target directory if it doesn't already exist
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        # Copy each CSV file to the target directory
        for csv_file_path in tables_path:
            if os.path.isfile(csv_file_path):
                # Determine the base filename to use in the target directory
                base_filename = os.path.basename(csv_file_path)
                target_file_path = os.path.join(target_dir, base_filename)
                if os.path.exists(target_file_path):
                    continue
                # Copy the file
                shutil.copy(csv_file_path, target_file_path)
                print(f"Copied '{csv_file_path}' to '{target_file_path}'")
            else:
                print(f"Warning: '{csv_file_path}' does not exist or is not a file")

    def encode_table(self, tables_path: List[str], batch_size=32, **kwargs):
        # Extract the dataset name from kwargs
        dataset_name = kwargs.get('dataset_name')
        if not dataset_name:
            dataset_name = f"dataset_{str(uuid.uuid4())}"
        self.dataset_name = dataset_name
        self.copy_tables_to_data_folder(tables_path, dataset_name)
        index_dataset(dataset_name)

    def encode_query(self, queries, batch_size=32, **kwargs):
        results = []
        for query in queries:
            results.append(run_query(query, self.dataset_name))
        return results


solo = SoloCustomModel(model_path="")
csv_paths = ["/home/jixy2012/carl/solo_testing/data/carlqa/tables_csv/names.csv"]
solo.encode_table(tables_path=csv_paths, dataset_name="testing_interface")
print(solo.encode_query(["how old is Linda Taylor?"]))