# convert all m4a to wav in the dataset
import yaml
import os
from pydub import AudioSegment

def count_m4a_files(path_to_convert):
    # count m4a files
    m4a_count = 0
    for file in os.listdir(path_to_convert):
        if file.split('.')[-1] == 'm4a':
            m4a_count += 1
    print(f'Found {m4a_count} m4a files in {path_to_convert}')

# convert to wav
def convert_m4a_to_wav(m4a_path, wav_path):
    try:
        audio = AudioSegment.from_file(m4a_path, format="m4a")
        audio.export(wav_path, format="wav")
    except Exception as e:
        print(f'Error converting {m4a_path} to {wav_path}: {e}')

# convert all m4a files to wav
def convert_all_m4a_to_wav(path_to_convert):
    for file in os.listdir(path_to_convert):
        if file.split('.')[-1] == 'm4a':
            m4a_path = os.path.join(path_to_convert, file)
            wav_path = os.path.join(path_to_convert, file.split('.')[0] + '.wav')
            # check if wav file already exists
            if os.path.exists(wav_path):
                print(f'{wav_path} already exists')
                continue
            print(f'Converting {m4a_path} to {wav_path}')
            convert_m4a_to_wav(m4a_path, wav_path)

def delete_m4a_files(path_to_convert):
    for file in os.listdir(path_to_convert):
        if file.split('.')[-1] == 'm4a':
            m4a_path = os.path.join(path_to_convert, file)
            os.system(f'rm -rf {m4a_path}')

with open('config.yaml') as f:
    config = yaml.safe_load(f)
dataset_path = os.path.join(config['paths']['dataset'])

specify_path = 'uncropped'
path_to_convert = os.path.join(dataset_path, specify_path)
count_m4a_files(path_to_convert)
# convert_all_m4a_to_wav(path_to_convert)