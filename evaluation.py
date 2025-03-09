## one-off script for gnerating training tags from the uncropped macaulay library dataset

import os
import yaml
import pandas as pd
from main import read_tags
with open('config.yaml') as f:
    config = yaml.safe_load(f)
dataset_path= os.path.join(config['paths']['dataset'], 'uncropped')
output_csv_path = os.path.join(config['paths']['dataset'], 'evaluation/evaluation.csv')

# def generate_tags_from_folders():
#     if os.path.exists(output_csv_path):
#         print(f'overwriting {output_csv_path}')
#         os.system(f'rm -rf {output_csv_path}')
#     to_add = {}

#     # print files in path
#     for path in os.listdir(dataset_path):
#         # if directory
#         if os.path.isdir(os.path.join(dataset_path, path)):
#             print(f'Folder: {path}')
#             if path=='NZBB':
#                 # add with species 'korimako' to datatags
#                 for file in os.listdir(os.path.join(dataset_path, path)):
#                     to_add[file] = {'species': 'korimako'}
#             elif path=='TUI':
#                 # add with species 'tui'
#                 for file in os.listdir(os.path.join(dataset_path, path)):
#                     to_add[file] = {'species': 'tui'}

#     # print
#     print('Files to add:')
#     for file, tags in to_add.items():
#         print(f'{file} - {tags}')
    
#     # write to csv
#     with open(output_csv_path, 'w') as f:
#         f.write('filename,species\n')
#         for file, tags in to_add.items():
#             f.write(f'{file},{tags["species"]}\n')


# def fix_tags():
#     # replace m4a with wav
#     actual_tags = os.path.join(dataset_path, 'tags.csv')
#     datatags = pd.read_csv(actual_tags)
#     datatags = datatags.set_index('filename').T.to_dict('dict')

#     # replace m4a with wav
#     to_remove = []
#     to_add = {}
#     for file, tags in datatags.items():
#         if file.split('.')[-1] == 'm4a':
#             to_remove.append(file)
#             to_add[file.split('.')[0] + '.wav'] = tags
#     for file in to_remove:
#         del datatags[file]
#     datatags.update(to_add)
#     with open(output_csv_path, 'w') as f:
#         f.write('filename,species\n')
#         for file, tags in datatags.items():
#             f.write(f'{file},{tags["species"]}\n')

# def check_tags():
#     datatags = read_tags(output_csv_path)
#     for file, tags in datatags.items():
#         if 'species' not in tags:
#             print(f'No species tag for {file}')
#         elif tags['species'] not in ['korimako', 'tui']:
#             print(f'Invalid species tag for {file}: {tags["species"]}')
#     for file in os.listdir(dataset_path):
#         if (file.split('.')[-1] not in ['wav', 'mp3']):
#             print(f'Invalid file extension for {file}')
#         if file.split('.')[0] not in datatags:
#             print(f'No tag for {file}')
#     print(f'Checked {len(datatags)} tags')
    
# print('starting')