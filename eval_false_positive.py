# this script does attention maps for eval

from pathlib import Path
import os
import torch
import yaml
from classifiers.kaytoo_small import BirdSoundModel, TrainingParameters, DefaultAudio, FilePaths, eval_real_life_attention_maps
from classifiers.colors import custom_color_maps, generate_rainbow_colors
import re

with open('classifiers/use_case.yaml') as f:
    use_case = yaml.safe_load(f)
    options = use_case
paths = FilePaths(options=options)
train_cfg = TrainingParameters(options=options)
audio_cfg = DefaultAudio()

results_dir = Path('classifiers/evaluation_results') / 'Exp_2559' / 'Results'
model_paths = [f for f in os.listdir(results_dir) if f.startswith('binary_classifier-epoch=') and f.endswith('.ckpt')]
for model_path in model_paths:
    print(model_path)
    # bird_model = BirdSoundModel(train_cfg, audio_cfg, paths, in_channels=3)
    # model_state_dict = torch.load(results_dir / model_path)
    # bird_model.load_state_dict(model_state_dict)
    # bird_model.eval()
    # eval_dir = paths.EVAL_DIR
# select the model with the highest rl_auc
model_path = max(model_paths, key=lambda x: float(re.search(r'rl_auc=(\d+\.\d+)', x).group(1)))

print(f"Loading model from {model_path}")
bird_model = BirdSoundModel(train_cfg, audio_cfg, paths, in_channels=3)
model_state_dict = torch.load(results_dir / model_path)

bird_model.load_state_dict(model_state_dict)
bird_model.eval()

contour_levels=[0.5,0.6,0.7,0.8,0.9]
contour_colors = generate_rainbow_colors(len(contour_levels)+1)
# remove purple
contour_colors = contour_colors[:-1]
# reverse order
contour_colors = contour_colors[::-1]
# contour_levels = []
# contour_colors=[]

# low_snr_idxs = [8,10,15,23]
low_snr_idxs = [15]
do_low_snr = True
for index_val in range(300):
    if do_low_snr:
        if index_val > len(low_snr_idxs)-1:
            break
        positive_idx = low_snr_idxs[index_val]
        negative_idx = low_snr_idxs[index_val]
    else:
        positive_idx = index_val
        negative_idx = index_val
    eval_real_life_attention_maps(
        bird_model,
        eval_dir=paths.EVAL_DIR,
        audio_cfg=audio_cfg,
        cmap_spec=custom_color_maps['dusk'],
        contour_levels=contour_levels,
        contour_colors=contour_colors,
        pos_idx=positive_idx,
        neg_idx=negative_idx,
        plot=True,
    )

# audio_file_path = 'classifiers/augmented_dataset/sound_files/0.wav'

# plot_single_attention_map(
#     bird_model,
#     audio_file_path=audio_file_path,
#     audio_cfg=audio_cfg,
#     cmap_spec=custom_color_maps['dusk'],
#     contour_levels=contour_levels,
#     contour_colors=contour_colors,
#     plot=True
# )