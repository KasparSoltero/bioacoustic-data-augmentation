# checking if i can recover the metrics from the results.csv file

import os
import pandas as pd

# fields:
# epoch 
# train/box_loss 
# train/cls_loss 
# train/dfi_loss
# metrics/precision(B)
# metrics/recall(B)
# metrics/mAP50(B)
# metrics/mAP50-95(B)
# val/box_loss 
# val/cls_loss
# val/dfl_loss
columns_to_plot = ['val/box_loss', 'val/cls_loss', 'val/dfl_loss', 'metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)']
# path eg:
# /Users/kaspar/Documents/ecoacoustics/detector/spectral_detector/runs/detect/train2801

lines = {}
root_directory = '/Users/kaspar/Documents/ecoacoustics/detector/spectral_detector/runs/detect'
for f in os.listdir(root_directory):
    # check if it is a directory
    if os.path.isdir(os.path.join(root_directory, f)):
        # check if results csv exists
        if not os.path.exists(os.path.join(root_directory, f, 'results.csv')):
            continue
        print(f)
        lines[f] = {}
        # find results.csv
        results = pd.read_csv(os.path.join(root_directory, f, 'results.csv')).rename(columns=lambda x: x.strip())
        lines[f]['epoch'] = results['epoch']
        for column in columns_to_plot:
            lines[f][column] = results[column]

# plot
import matplotlib.pyplot as plt
for column in columns_to_plot:
    plt.figure()
    for key in lines.keys():
        plt.plot(lines[key]['epoch'], lines[key][column], label=key)
    plt.xlabel('epoch')
    plt.ylabel(column)
    plt.legend()
    plt.title(column)
plt.show()