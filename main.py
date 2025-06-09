import json
import yaml
import os
import torchaudio
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import hsv_to_rgb
from PIL import Image
import csv
import chardet
from matplotlib.ticker import PercentFormatter
from skimage.measure import find_contours, approximate_polygon
from skimage.morphology import opening, closing, disk
from scipy.spatial import ConvexHull
import matplotlib.patches as patches

from spectrogram_tools import spectrogram_transformed, spec_to_audio, crop_overlay_waveform, load_waveform, transform_waveform, map_frequency_to_log_scale, map_frequency_to_linear_scale, merge_boxes_by_class, generate_masks, rle_decode, rle_encode, log_scale_spectrogram # Added rle_encode, log_scale_spectrogram
from classifiers.colors import custom_color_maps, hex_to_rgb

def generate_yolo_segment_data_from_binary_mask(binary_mask, class_id, box_in_full_spec_bins, full_spec_dims_bins, simplify_tolerance=10, морph_footprint_size=1, min_contour_area_pixels=100): # Increased min_contour_area_pixels default
    """
    Generates YOLO segmentation data from a binary mask.
    Args:
        binary_mask (np.array): The 2D binary mask for the object.
                                **IMPORTANT ASSUMPTION: This mask is the size of the FULL spectrogram,
                                containing only the target vocalization.**
        class_id (int): The class ID of the object.
        box_in_full_spec_bins (list): [t_min, t_max, f_min, f_max] of the vocalization's nominal bounding box
                                      within the full spectrogram. Currently NOT USED for scaling in this revised version,
                                      but could be used for pre-cropping binary_mask or filtering contours.
        full_spec_dims_bins (tuple): (total_time_bins_full, total_freq_bins_full) of the full spectrogram.
        simplify_tolerance (float): Tolerance for polygon simplification.
        morph_footprint_size (int): Size of the footprint for morphological operations.
        min_contour_area_pixels (int): Minimum pixel area for a contour to be kept.
    Returns:
        list: A list of strings, each formatted for YOLO segmentation.
    """
    
    cleaned_mask = binary_mask.copy() 

    if морph_footprint_size > 0:
        selem = disk(морph_footprint_size) 
        cleaned_mask = opening(cleaned_mask, selem)
        cleaned_mask = closing(cleaned_mask, selem)

    contours = find_contours(cleaned_mask, 0.5) 
    yolo_strings = []

    total_time_bins_full, total_freq_bins_full = full_spec_dims_bins
    
    for contour in contours:
        if contour.shape[0] < 3 : 
            continue
            
        r_coords = contour[:, 0] # row indices (frequency)
        c_coords = contour[:, 1] # col indices (time)
        
        # Approximate area using bounding box of the contour on the (potentially full-size) mask
        contour_height_pixels = np.max(r_coords) - np.min(r_coords) + 1
        contour_width_pixels = np.max(c_coords) - np.min(c_coords) + 1
        
        if contour_height_pixels * contour_width_pixels < min_contour_area_pixels:
             continue

        simplified_contour = approximate_polygon(contour, tolerance=simplify_tolerance)

        if len(simplified_contour) < 3: 
            continue

        points_str_list = [f"{class_id}"]
        for point in simplified_contour:
            # point[0] is row (frequency bin index from top of mask array)
            # point[1] is col (time bin index from left of mask array)
            # These are ALREADY absolute bin indices if cleaned_mask is full-size.
            abs_f_bin, abs_t_bin = point[0], point[1]
            
            # Directly normalize these absolute bin coordinates
            x_yolo = np.clip(abs_t_bin / (total_time_bins_full -1 if total_time_bins_full > 1 else 1) , 0.0, 1.0)
            
            # Determine y_yolo based on mask's frequency axis orientation:
            # Spectrograms from torchaudio: freq[0] = low freq.
            # find_contours: point[0] (row index) starts at 0 for top row of array.
            # If binary_mask[0,:] is lowest frequency (bottom of spectrogram):
            #   abs_f_bin is distance from bottom of spectrogram in mask rows.
            #   Normalized from bottom: abs_f_bin / (total_freq_bins_full-1)
            #   YOLO Y (0 at top): 1.0 - (normalized from bottom)
            # If binary_mask[0,:] is highest frequency (top of spectrogram):
            #   abs_f_bin is distance from top of spectrogram in mask rows.
            #   YOLO Y: abs_f_bin / (total_freq_bins_full-1)

            # Based on your `generate_masks` and how `binary_mask_overlay` is created,
            # `binary_mask_overlay[0,:]` corresponds to the lowest frequency band if derived
            # directly from a standard spectrogram array [freq, time] where freq[0] is low.
            # `find_contours` treats row 0 as the "top" of this array.
            # So, `abs_f_bin` (row index) increasing means higher frequency.
            
            y_yolo_raw_from_bottom = abs_f_bin / (total_freq_bins_full - 1 if total_freq_bins_full > 1 else 1)
            y_yolo = np.clip(1.0 - y_yolo_raw_from_bottom, 0.0, 1.0)

            points_str_list.append(f"{x_yolo:.5f}")
            points_str_list.append(f"{y_yolo:.5f}")
        
        # Ensure we still have at least 3 unique points for a valid polygon
        # (This check might be implicitly handled by len(simplified_contour) < 3 earlier)
        # For robustness, check number of coordinate pairs
        if (len(points_str_list) -1 ) / 2 >= 3: # -1 for class_id, /2 for xy pairs
            yolo_strings.append(" ".join(points_str_list))
            
    return yolo_strings

def plot_labels(config, idx=[0,-1], save_directory='output'):
    if not config['output']['include_spectrogram']:
        print('Spectrograms are not included in the output; skipping plot_labels')
        return

    # Load species_value_map for class names if available
    species_value_map = {}
    species_map_path = f'{save_directory}/species_value_map.csv'
    if os.path.exists(species_map_path):
        with open(species_map_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) == 2:
                    species_value_map[int(row[0])] = row[1]


    # Calculate the number of rows and columns for subplots
    if idx[1] == -1: # Default to plotting first 9 images if end index is -1
        idx[1] = min(9, len(os.listdir(f'{save_directory}/artificial_dataset/images')))
    
    num_images_to_plot = idx[1] - idx[0]
    if num_images_to_plot <= 0:
        print("No images to plot in the specified range.")
        return

    cols = 3 # Fixed number of columns
    # Calculate base rows needed for spectrograms
    base_rows = (num_images_to_plot + cols - 1) // cols 

    actual_figure_rows = base_rows
    masks_enabled = config['output']['include_yolo_masks'] or config['output']['include_coco_masks']
    if masks_enabled:
        actual_figure_rows = base_rows * 2 # Double rows if masks are to be plotted underneath

    # Limit rows to prevent excessively large plots, e.g., max 4 image rows (8 total rows with masks)
    MAX_IMAGE_ROWS = 4
    if base_rows > MAX_IMAGE_ROWS:
        print(f"Limiting plot to {MAX_IMAGE_ROWS*cols} images due to too many rows.")
        num_images_to_plot = MAX_IMAGE_ROWS * cols
        base_rows = MAX_IMAGE_ROWS
        if masks_enabled:
            actual_figure_rows = base_rows * 2
        else:
            actual_figure_rows = base_rows
        idx[1] = idx[0] + num_images_to_plot


    if actual_figure_rows == 0:
        print("Calculated 0 rows for plotting. Exiting plot_labels.")
        return

    fig, axes = plt.subplots(actual_figure_rows, cols, figsize=(7.5*cols, 4.5 * actual_figure_rows), squeeze=False)
    # `squeeze=False` ensures axes is always 2D, even for 1 row/col
    fig.canvas.manager.set_window_title('') 
    fig.suptitle(f'{save_directory}/artificial_dataset/images (Displaying {num_images_to_plot} images)', fontsize=12)


    image_files = [f for f in os.listdir(f'{save_directory}/artificial_dataset/images') if not f.startswith('.')]
    # Sort files to ensure consistent plotting order, especially if OS doesn't guarantee it
    # Assuming filenames like "0.jpg", "1.jpg" or "train/0.jpg"
    try:
        # Attempt to sort numerically if filenames are like "N.jpg" or "prefix/N.jpg"
        image_files.sort(key=lambda f: int(os.path.splitext(os.path.basename(f))[0]))
    except ValueError:
        image_files.sort() # Fallback to lexicographical sort

    for i, image_path_basename in enumerate(image_files[idx[0]:idx[1]]):
        # Determine subplot indices for spectrogram and (potentially) mask
        # This assumes image_path_basename is just "0.jpg", "1.jpg" etc.
        # If image_path_basename includes "train/" or "val/", this logic needs adjustment
        # For this loop, `i` is the 0-based index within the selected slice of images.

        current_image_set_row = i // cols
        current_col_idx = i % cols

        # Define axes for spectrogram
        spec_ax_row = current_image_set_row
        if masks_enabled:
            spec_ax_row = current_image_set_row * 2
        
        if spec_ax_row >= axes.shape[0] or current_col_idx >= axes.shape[1]:
            print(f"Plotting stopped at image {i}, subplot index out of bounds.")
            break
        
        ax = axes[spec_ax_row][current_col_idx]
        
        full_image_path = os.path.join(f'{save_directory}/artificial_dataset/images', image_path_basename)
        try:
            image = Image.open(full_image_path)
        except FileNotFoundError:
            print(f"Image file not found: {full_image_path}")
            ax.text(0.5, 0.5, 'Image not found', ha='center', va='center')
            ax.set_xticks([])
            ax.set_yticks([])
            continue
            
        image_array = np.array(image)
        img_height, img_width = image_array.shape[0], image_array.shape[1] # Usually (height, width, channels)

        # Plot bounding boxes if enabled
        if config['output']['include_boxes']:
            print(f"Plotting bounding boxes for {image_path_basename}")
            # Construct label path: image_path_basename might be "0.jpg" or "train/0.jpg" etc.
            # So, label_path should be relative to box_labels directory
            label_file_name = os.path.splitext(image_path_basename)[0] + ".txt"
            label_path = f'{save_directory}/artificial_dataset/box_labels/{label_file_name}'
            
            if os.path.exists(label_path):
                boxes = []
                with open(label_path, 'r') as f:
                    for line in f:
                        values = [value.strip() for value in line.split(' ')]
                        if len(values) == 5:
                            class_id, x_center, y_center, width, height = [float(value) for value in values if value]
                            boxes.append([class_id, x_center, y_center, width, height])
    
                for box_data in boxes:
                    class_id, x_center, y_center, width, height = box_data
                    # Convert YOLO box format (normalized, center_x, center_y, w, h) to pixel coordinates for plotting
                    y_center = 1.0 - y_center # YOLO y_center is from bottom, convert to top
                    # (0,0) for YOLO box is top-left of image. Matplotlib imshow also has (0,0) at top-left by default (origin='upper')
                    box_time_width = width * 10
                    box_freq_height = height * 24000
                    x_min_time = (x_center * 10) - (box_time_width / 2)
                    y_min_freq = (y_center * 24000) - (box_freq_height / 2) # y_center is from top

                    rect = plt.Rectangle((x_min_time, y_min_freq), box_time_width, box_freq_height,
                                        linewidth=1, edgecolor='#FFFFFF', facecolor='none', alpha=0.8)
                    ax.add_patch(rect)
                    if species_value_map and config['plot']['show_labels']:
                        labeltext = species_value_map.get(int(class_id), f"ID:{int(class_id)}")
                        if ' ' in labeltext and len(labeltext) > 10: # Basic wrap
                            labeltext = labeltext.split(' ')[0] + '\n' + ' '.join(labeltext.split(' ')[1:])
                        ax.text(x_min_time, y_min_freq + box_freq_height+10, labeltext, fontsize=10, color='#eeeeee',
                                bbox=dict(facecolor='black', alpha=0.3, pad=0.8, edgecolor='none'))

        # Plot the spectrogram image
        cmap_to_use = None
        if config['plot']['color_filter'] == 'dusk':
            cmap_to_use = custom_color_maps['dusk']
        elif not config['output']['rainbow_frequency']: # If not rainbow, use gray
            cmap_to_use = 'gray'
        # If rainbow_frequency is true and no dusk filter, cmap_to_use remains None (default matplotlib cmap)

        
        # Spectrogram Y-axis ticks (frequency)
        # Assuming image_height corresponds to max_freq (e.g., 24000 Hz log-scaled)
        # And image_array y-coords are 0 (top, high_freq_log) to img_height (bottom, low_freq_log)
        log_yticks_pixel = map_frequency_to_log_scale(img_height, [0, 1000, 2000, 5000, 10000, 24000])
        ax.set_yticks(log_yticks_pixel)
        yticklabels = ['0', '1', '2', '5', '10', '24']
        ax.set_yticklabels(yticklabels)
        ax.imshow(image_array, aspect='auto', origin='upper', extent=[0, 10, 0, 24000], cmap=cmap_to_use) # img_height for extent ymax
        
        ax.set_title(f'{os.path.basename(image_path_basename)}', fontsize=9)
        if current_image_set_row == base_rows -1 or (masks_enabled and spec_ax_row == actual_figure_rows -2): # Show X labels only on bottom-most spec row
            ax.set_xlabel('Time (s)', fontsize=10)
        else:
            ax.set_xticklabels([])
        ax.set_ylabel('Freq (kHz)', fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=8)

        # Plot MASKS (YOLO or COCO) on the subplot below the spectrogram
        if masks_enabled:
            mask_ax_row = spec_ax_row + 1
            if mask_ax_row >= axes.shape[0] or current_col_idx >= axes.shape[1]:
                print(f"Cannot plot mask for image {i}, subplot index out of bounds for mask.")
                continue # Skip mask plotting for this image

            thisax = axes[mask_ax_row][current_col_idx]
            thisax.clear()
            thisax.set_xticks([])
            thisax.set_yticks([])
            # Display a light version of the original image as background for masks
            thisax.imshow(image_array, aspect='auto', origin='upper', alpha=1, cmap=cmap_to_use)


            if config['output']['include_yolo_masks']:
                yolo_label_filename = os.path.splitext(image_path_basename)[0] + ".txt"
                yolo_label_path = f'{save_directory}/artificial_dataset/yolo_labels/{yolo_label_filename}'

                if os.path.exists(yolo_label_path):
                    polygons_to_draw = []
                    with open(yolo_label_path, 'r') as f_yolo:
                        for line_num, line in enumerate(f_yolo):
                            parts = line.strip().split()
                            if not parts: continue
                            try:
                                class_id_yolo = int(parts[0])
                                points_norm = [float(p) for p in parts[1:]]
                                if len(points_norm) < 6 or len(points_norm) % 2 != 0: # Min 3 points (6 values)
                                    # print(f"Skipping invalid YOLO line in {yolo_label_path}: {line.strip()}")
                                    continue
                            except ValueError:
                                # print(f"Skipping malformed YOLO line in {yolo_label_path}: {line.strip()}")
                                continue
                            
                            polygon_points_pixel = []
                            for k_coord in range(0, len(points_norm), 2):
                                x_norm, y_norm = points_norm[k_coord], points_norm[k_coord+1]
                                x_pixel = x_norm * img_width
                                y_pixel = y_norm * img_height 
                                polygon_points_pixel.append([x_pixel, y_pixel])
                            
                            if polygon_points_pixel:
                                polygons_to_draw.append({'class_id': class_id_yolo, 'points': polygon_points_pixel})
                    
                    if polygons_to_draw:
                        thisax.set_title(f'YOLO Masks ({len(polygons_to_draw)})', fontsize=8)
                        for poly_idx, poly_data in enumerate(polygons_to_draw):
                            # Use class_id for color if a map exists, otherwise cycle
                            color_hex = custom_color_maps['rotary'][poly_data['class_id'] % len(custom_color_maps['rotary'])]
                            # color_hex = custom_color_maps['rotary'][poly_idx % len(custom_color_maps['rotary'])] # Color by instance
                            
                            polygon_patch = patches.Polygon(poly_data['points'], closed=True, fill=True, 
                                                            edgecolor='white', facecolor=tuple(c/255.0 for c in hex_to_rgb(color_hex)) + (0.6,),# Add alpha to facecolor
                                                            linewidth=1)
                            thisax.add_patch(polygon_patch)
                    else:
                        thisax.text(0.5, 0.5, 'YOLO file empty or invalid', ha='center', va='center', transform=thisax.transAxes, fontsize=8)
                        thisax.set_title(f'YOLO Masks (empty)', fontsize=8)

                else: # YOLO file does not exist
                    thisax.text(0.5, 0.5, 'No YOLO .txt', ha='center', va='center', transform=thisax.transAxes, fontsize=8)
                    thisax.set_title(f'YOLO Masks (no file)', fontsize=8)

            elif config['output']['include_coco_masks']: # Fallback to COCO if YOLO not enabled
                coco_path = f'{save_directory}/artificial_dataset/mask_annotations.json'
                if os.path.exists(coco_path):
                    with open(coco_path, 'r') as f:
                        coco_data = json.load(f)
                    
                    image_name_stem_for_coco = os.path.splitext(image_path_basename)[0]
                    image_id_coco = None
                    for img_coco_dict in coco_data['images']:
                        coco_file_stem = os.path.splitext(os.path.basename(img_coco_dict['file_name']))[0]
                        if coco_file_stem == image_name_stem_for_coco: # Match stem, e.g., "0"
                            # Further check if full path matches if prefixes are used
                            if os.path.basename(img_coco_dict['file_name']) == image_path_basename:
                                image_id_coco = img_coco_dict['id']
                                break
                            # If no direct basename match, take the first stem match (less precise)
                            if image_id_coco is None: image_id_coco = img_coco_dict['id'] 
                    
                    # if image_id_coco is None and image_name_stem_for_coco.isdigit(): # Try direct ID if stem is numeric
                    #     image_id_coco = int(image_name_stem_for_coco)


                    if image_id_coco is not None:
                        image_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id_coco]
                        
                        if image_annotations:
                            thisax.set_title(f'COCO Masks ({len(image_annotations)})', fontsize=8)
                            # Create a composite mask overlay image
                            # Using the alpha blended background already set by thisax.imshow(image_array, alpha=0.4)
                            # We will draw patches on top of this.

                            for j, ann in enumerate(image_annotations):
                                if 'counts' not in ann['segmentation'] or 'size' not in ann['segmentation']:
                                    continue
                                mask_counts = ann['segmentation']['counts']
                                mask_size = ann['segmentation']['size'] # [orig_freq_bins, orig_time_bins]
                                decoded_mask = rle_decode(mask_counts, mask_size) # 1D array
                                decoded_mask_2d = decoded_mask.reshape(mask_size) # Reshape to 2D [freq, time]

                                # Resize this mask to the display image dimensions
                                mask_resized_pil = Image.fromarray(decoded_mask_2d.astype(np.uint8) * 255).resize(
                                    (img_width, img_height), Image.NEAREST
                                )
                                mask_resized_np = np.array(mask_resized_pil)
                                mask_resized_np = np.flipud(mask_resized_np)
                                
                                # here
                                contours_coco = find_contours(mask_resized_np, 128) # Find contours on the resized mask
                                color_coco_hex = custom_color_maps['rotary'][j % len(custom_color_maps['rotary'])]
                                for contour in contours_coco:
                                    # Contour points are (row, col) which is (y, x)
                                    # Patches.Polygon expects (x, y)
                                    polygon_coco_patch = patches.Polygon(contour[:, [1, 0]], closed=True, fill=True,
                                                                        edgecolor='white', facecolor=tuple(c/255.0 for c in hex_to_rgb(color_coco_hex)) + (0.6,), # Add alpha to facecolor
                                                                        linewidth=1)
                                    thisax.add_patch(polygon_coco_patch)
                        else: # No annotations for this image_id
                            thisax.text(0.5, 0.5, 'COCO: No annotations for ID', ha='center', va='center', transform=thisax.transAxes, fontsize=8)
                            thisax.set_title(f'COCO Masks (no anns)', fontsize=8)
                    else: # Image ID not found in COCO json
                        thisax.text(0.5, 0.5, 'COCO: Image ID not found', ha='center', va='center', transform=thisax.transAxes, fontsize=8)
                        thisax.set_title(f'COCO Masks (no ID)', fontsize=8)
                else: # COCO json file not found
                    thisax.text(0.5, 0.5, 'COCO .json not found', ha='center', va='center', transform=thisax.transAxes, fontsize=8)
                    thisax.set_title(f'COCO Masks (no json)', fontsize=8)
            else: # No mask type enabled, or problem with axes
                thisax.text(0.5, 0.5, 'Masks not configured', ha='center', va='center', transform=thisax.transAxes, fontsize=8)


    plt.tight_layout(rect=[0, 0, 1, 0.97]) # Adjust rect to make space for suptitle
    plt.show()
    plt.close()

def _plot_labels(config, idx=[0,-1], save_directory='output'):
    if not config['output']['include_spectrogram']:
        print('Spectrograms are not included in the output; skipping plot_labels')
        return
    # Plotting the labelsq
    # check if species value map exists
    species_value_map = {}
    if os.path.exists(f'{save_directory}/species_value_map.csv'):
        with open(f'{save_directory}/species_value_map.csv', 'r') as f:
            for line in f:
                key, value = line.strip().split(',') #reading in reverse
                species_value_map[int(key)] =value
    # Plotting the spectrograms
    # Calculate the number of rows needed
    if idx[1] == -1:
        idx[1] = 9
    rows = (idx[1] - idx[0]) //  3 if (idx[1] - idx[0]) else 1
    if rows < 1:
        rows = 1
    if config['output']['include_yolo_masks'] or config['output']['include_coco_masks']:
        rows *= 2
    if rows > 4: # Limit rows to prevent excessively large plots
        rows = 4
    
    if idx[1] - idx[0] < 3:
        cols = (idx[1] - idx[0]) % 3
    else:
        cols = 3

    # Plotting the spectrograms
    fig, axes = plt.subplots(rows, cols, figsize=(7.5*cols, 4.5 * rows))
    fig.canvas.manager.set_window_title('') 
    fig.suptitle(f'{save_directory}/artificial_dataset/images', fontsize=12)

    # Ensure axes is always a 2D array
    axes = np.array(axes).reshape(rows, -1)

    for i, image_path in enumerate(os.listdir(f'{save_directory}/artificial_dataset/images')[idx[0]:idx[1]]):
        if image_path == '.DS_Store':
            continue
        
        # Compute row and column index
        row_idx = i // 3
        col_idx = i % 3
        if row_idx >= rows:
            print(f'Plotting stopped at {i} images - too many rows')
            break
        if col_idx >= cols:
            print(f'Plotting stopped at {i} images - too many columns')
            break
        
        image = Image.open(f'{save_directory}/artificial_dataset/images/{image_path}')
        ax = axes[row_idx][col_idx]
        image_array = np.array(image)

        if config['output']['include_boxes']:
            label_path = f'{save_directory}/artificial_dataset/box_labels/{image_path[:-4]}.txt'
            # get the corresponding label
            boxes = []
            with open(label_path, 'r') as f:
                for line in f:
                    # Split on commas and strip whitespace
                    # values = [value.strip() for value in line.split(',')]
                    # values separated by spaces
                    values = [value.strip() for value in line.split(' ')]
                    
                    # Convert to float, ignoring empty strings
                    class_id, x_center, y_center, width, height = [float(value) for value in values if value]
                    
                    boxes.append([class_id, x_center, y_center, width, height])
    
            # plot boxes
            for box in boxes:
                x_center, y_center, width, height = box[1:]
                x_min = x_center * 10  # Multiply by 10 to match the time axis
                y_min = (1 - y_center) * 24000  # Adjust y-coordinate for upper origin
                box_width = width * 10
                box_height = height * 24000
                rect = plt.Rectangle((x_min - box_width/2, y_min - box_height/2), box_width, box_height,
                                    linewidth=1, edgecolor='#ffffff', facecolor='none')
                # elif box[0] == 0:
                    # rect = plt.Rectangle((x_min - box_width/2, y_min - box_height/2), box_width, box_height, 
                                        # linewidth=1, edgecolor='white', facecolor='none', linestyle='--')
                # elif box[0] == 1:
                    # rect = plt.Rectangle((x_min - box_width/2, y_min - box_height/2), box_width, box_height, 
                                        # linewidth=1, edgecolor='r', facecolor='none', linestyle='--')
                ax.add_patch(rect)
                if species_value_map and config['plot']['show_labels']:
                    labeltext = species_value_map[int(box[0])]
                    # insert newlines
                    if ' ' in labeltext:
                        labeltext = labeltext.split(' ')[0] + '\n' + labeltext.split(' ')[1]
                    ax.text(x_min + box_width/2, y_min + box_height/2, labeltext, fontsize=6, color='#eeeeee')

        if config['output']['include_coco_masks']:
            # Load COCO annotations
            coco_path = f'{save_directory}/artificial_dataset/mask_annotations.json'
            if os.path.exists(coco_path):
                with open(coco_path, 'r') as f:
                    coco_data = json.load(f)
                
                # Find annotations for current image
                image_name = image_path[:-4]  # Remove .jpg extension
                image_id = None
                for img in coco_data['images']:
                    if img['file_name'].startswith(image_name):
                        image_id = img['id']
                        break
                
                if image_id is not None:
                    # Get all annotations for this image
                    image_annotations = [ann for ann in coco_data['annotations'] 
                                      if ann['image_id'] == image_id]
                    print(f'found {len(image_annotations)} annotations for {image_name}')
                    
                    # Create a colored mask overlay
                    mask_overlay = np.zeros_like(image_array)
                    if len(mask_overlay.shape) != 3:
                        # add 3 channels
                        mask_overlay = np.stack([mask_overlay, mask_overlay, mask_overlay], axis=-1)
                    
                    for j, ann in enumerate(image_annotations):
                        mask_counts = ann['segmentation']['counts']
                        mask_size = ann['segmentation']['size']
                        mask = rle_decode(mask_counts, mask_size)
                        
                        mask = mask.reshape(mask_size)  # Should be [freq, time]

                        # Convert mask to image size maintaining aspect ratio
                        freq_bins, time_bins = mask.shape
                        scale_freq = image_array.shape[0]/freq_bins
                        scale_time = image_array.shape[1]/time_bins
                        new_freq = int(freq_bins * scale_freq)
                        new_time = int(time_bins * scale_time)

                        mask_resized = np.array(Image.fromarray(mask.astype(np.uint8) * 255).resize(
                            (new_time, new_freq), 
                            Image.NEAREST
                        ))
                        
                        color = hex_to_rgb(custom_color_maps['rotary'][j % len(custom_color_maps['rotary'])])
                        mask_overlay[mask_resized > 0] = color
                    # invert y axis
                    mask_overlay = np.flipud(mask_overlay)

            # plot the masks on the next axes
            thisax = axes[row_idx+1][col_idx]
            thisax.imshow(mask_overlay, aspect='auto', origin='upper')
            thisax.set_xticks([])
            thisax.set_yticks([])
        #     # sum arrays
        #     # image_array = image_array + mask_overlay
        #     image_array = mask_overlay

        # Display image with mask overlay
        if config['plot']['color_filter'] == 'dusk':
            im = ax.imshow(image_array, aspect='auto', origin='upper', extent=[0, 10, 0, 24000], cmap=custom_color_maps['dusk'])
        else:
            if config['output']['rainbow_frequency']:
                im = ax.imshow(image_array, aspect='auto', origin='upper', extent=[0, 10, 0, 24000])
            else:
                im = ax.imshow(image_array, aspect='auto', origin='upper', extent=[0, 10, 0, 24000], cmap='gray')

        yticks = [0, 1000, 2000, 5000, 10000, 24000]
        logyticks = map_frequency_to_log_scale(24000, yticks)
        ax.set_yticks(logyticks)
        yticklabels = [0, 1, 2, 5, 10, 24]
        ax.set_yticklabels(yticklabels)
        ax.set_title(f'{image_path[:1]}')
        ax.set_xlabel('Time (s)', fontsize=18)
        ax.set_ylabel('Frequency (kHz)', fontsize=18)
        # axis tick font size
        ax.tick_params(axis='both', which='major', labelsize=18-2)
    
    plt.tight_layout()
    plt.show()
    plt.close()

def read_tags(path, config, default_species='unknown'):
    # reads a csv, returns dictionaries of filenames with each column's attributes
    tags_path = os.path.join(path, 'tags.csv')
    if os.path.exists(tags_path):
        with open(tags_path, 'rb') as raw_file:
            result = chardet.detect(raw_file.read())
            encoding = result['encoding']
        
        with open(tags_path, mode='r', newline='', encoding=encoding) as file:
            reader = csv.DictReader(file)
            tags_data = {}
            for row in reader:
                filename = row['filename']
                filename = filename.split('.')[:-1]
                filename = '.'.join(filename)
                tags_data[filename] = {}
                for header in reader.fieldnames:
                    tags_data[filename][header] = row[header]
                
        return tags_data
    else:
        tags_data = {}
        for f in os.listdir(path):
            for ext in config['input']['allowed_files']:
                if f.endswith(ext) and not f.startswith('.'):
                    found_filename = f.split('.')[0]
                    tags_data[found_filename] = {'filename': found_filename, 'species': default_species}
        return tags_data

def load_input_dataset(data_root, background_path, positive_path, negative_path, config):
    positive_segment_paths = []
    positive_datatags = read_tags(os.path.join(data_root, positive_path), config, 1)
    
    # Get the list of files and randomly shuffle it
    files = os.listdir(os.path.join(data_root, positive_path))
    random.shuffle(files)
    for f in files:
        if config['input']['limit_positives'] and len(positive_segment_paths) >= config['input']['limit_positives']:
            print(f'Limiting positive examples to {config["input"]["limit_positives"]}')
            print(f'chosen positives: {positive_segment_paths}')
            break

        for ext in config['input']['allowed_files']:
            if f.endswith(ext) and not f.startswith('.'):
                found_filename = f.split('.')[:-1]
                found_filename = '.'.join(found_filename)
                #  overlay_label for tracing training data, e.g. 5th bird -> bi5
                if found_filename in positive_datatags:
                    positive_datatags[found_filename]['overlay_label'] = positive_path[:2]+str(list(positive_datatags.keys()).index(found_filename))
                else: 
                    positive_datatags[found_filename] = {'filename': found_filename, 'species': 'unknown', 'overlay_label': 'unknown'}
                full_audio_path = os.path.join(data_root, positive_path, f)

                # temp #todo remove just quickly
                if positive_datatags[found_filename]['species'] == 'unknown':
                    break # skip unknown species

                positive_segment_paths.append(full_audio_path)
                break

    negative_segment_paths = []
    negative_datatags = read_tags(os.path.join(data_root, negative_path), config, 0)
    for f in os.listdir(os.path.join(data_root, negative_path)):
        for ext in config['input']['allowed_files']:
            if f.endswith(ext) and not f.startswith('.'):
                found_filename = f.split('.')[:-1]
                found_filename = '.'.join(found_filename)
                negative_datatags[found_filename]['overlay_label'] = negative_path[:2]+str(list(negative_datatags.keys()).index(found_filename))
                negative_segment_paths.append(os.path.join(data_root, negative_path, f))
                break

    background_noise_paths = []
    background_datatags = read_tags(os.path.join(data_root, background_path), config)
    for f in os.listdir(os.path.join(data_root, background_path)):
        for ext in config['input']['allowed_files']:
            if f.endswith(ext) and not f.startswith('.'):
                found_filename = f.split('.')[:-1]
                found_filename = '.'.join(found_filename)
                background_datatags[found_filename]['overlay_label'] = 'bg'+str(list(background_datatags.keys()).index(found_filename))
                background_noise_paths.append(os.path.join(data_root, background_path, f))
                break

    return positive_segment_paths, positive_datatags, negative_segment_paths, negative_datatags, background_noise_paths, background_datatags

def write_species_value_map_to_file(species_value_map, save_directory='output'):
    # write the species value map to a file
    with open(f'{save_directory}/species_value_map.csv', 'w') as f:
        for key, value in species_value_map.items():
            f.write(f'{value},{key}\n') # reverse order for easy reading

def find_max_index(directory):
    """Find the maximum index from existing files in a directory"""
    max_idx = -1
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            if filename.endswith('.wav'):
                try:
                    idx = int(filename.split('.')[0])
                    max_idx = max(max_idx, idx)
                except ValueError:
                    continue
    return max_idx

def generate_overlays(
        config,
        get_data_paths=[None, None, None, None],
        save_directory='datasets_mutable',
        n=1,
        sample_rate=48000,
        final_length_seconds=10,
        positive_overlay_range=[1,1],
        negative_overlay_range=[0,0],
        save_wav=False,
        plot=False,
        clear_dataset=False,
        val_ratio = 0.8,
        snr_range=[0.1,1],
        repetitions=[1,10],
        specify_positive=None,
        specify_noise='/Volumes/Rectangle/bioacoustic-data-augmentation-dataset/noise/Heavy-Rain-Falling-Off-Roof-A1-www.fesliyanstudios.com_01.wav',
        specify_bandpass=None,
        color_mode='HSV'
    ):
    # Loop for creating and overlaying spectrograms
    # DEFAULTS: 
        # noise normalised to 1 rms, dB
        # song set to localised snr 1-10
        # song bbox threshold 5 dB over 10 bands (240hz)
        # songs can be cropped over edges, minimum 1 second present
        # images are normalised to 0-100 dB, then 0-1 to 255
        # 80:20 split train and val
        # 640x640 images
    # TODO:
        # training data spacings for long ones, add distance/spacing random additions in loop

    # Determine starting index for concatenation mode
    start_idx = 0
    if config['output']['concatenate']:
        sound_files_dir = 'classifiers/augmented_dataset/sound_files'
        start_idx = find_max_index(sound_files_dir) + 1
        print(f"Concatenating from index {start_idx}")
    if config['input']['limit_positives'] and config['input']['limit_positives'] > 0:
        # ensure positive_overlay_range is not greater than limit_positives
        positive_overlay_range[1] = min(positive_overlay_range[1], config['input']['limit_positives'])

    if clear_dataset and not config['output']['concatenate']:
        os.system(f'rm -rf {save_directory}/artificial_dataset/images/*')
        os.system(f'rm -rf {save_directory}/artificial_dataset/box_labels/*')
        os.system(f'rm -rf {save_directory}/artificial_dataset/yolo_labels/*') # Clear YOLO labels
        os.system(f'rm -rf {save_directory}/artificial_dataset/dataset.yaml') # Clear YOLO dataset yaml
        os.system(f'rm -rf {save_directory}/artificial_dataset/mask_annotations.json') # Clear COCO annotations
        os.system(f'rm -rf {save_directory}/species_value_map.csv')
        os.system(f'rm -rf classifiers/augmented_dataset/sound_files/*')
        os.system(f'rm -rf classifiers/augmented_dataset/labels.csv')

    data_root, background_path, positive_paths, negative_paths = get_data_paths
    if data_root is None:
        data_root='../data/manually_isolated'
        background_path='background_noise'
    if positive_paths is None:
        positive_paths = ['unknown', 'amphibian', 'reptile', 'mammal', 'insect', 'bird']
        negative_paths = ['anthrophony', 'geophony']
    positive_segment_paths, positive_datatags, negative_segment_paths, negative_datatags, background_noise_paths, background_datatags = load_input_dataset(data_root, background_path, positive_paths, negative_paths, config)

    # load config
    # generate species value map
    species_value_map = {}
    yolo_data_for_files = {}
    coco_annotations = []

    if config['output']['include_yolo_masks']:
        print("YOLO mask output enabled.")
        # species_value_map will be populated and used for dataset.yaml
        # yolo_data_for_files = {} # Already initialized above
    elif config['output']['include_coco_masks']:
        print("COCO mask output enabled.")
        coco_dataset = {
            'images': [],
            'annotations': [],
            'categories': []
        }
    # No specific init for yolo_data_for_files here, it's populated per image.
        
    if config['output']['include_kaytoo']:
        # Initialize the file with headers ONCE before the loop
        labels_path = f'classifiers/augmented_dataset/labels.csv'
        if not os.path.exists(labels_path):
            os.makedirs(os.path.dirname(labels_path), exist_ok=True)
            with open(labels_path, 'w') as f:
                f.write('filename,primary_label\n')

    val_index = int(n*val_ratio) # validation

    # main loop to create and overlay audio
    for idx_offset in range(n):
        idx = start_idx + idx_offset  # Use adjusted index
        if idx == n:
            break
        # label = str(idx) # image label
        # Select a random background noise (keep trying until one is long enough)
        noise_db = -9
        bg_noise_waveform_cropped = None
        while bg_noise_waveform_cropped is None:
            if specify_noise is not None:
                bg_noise_path = specify_noise
            else:
                bg_noise_path = random.choice(background_noise_paths)
            bg_noise_waveform, original_sample_rate = load_waveform(bg_noise_path)
            bg_noise_waveform_cropped = transform_waveform(bg_noise_waveform, 
                resample=[original_sample_rate,sample_rate], 
                random_crop_seconds=final_length_seconds
            )
        if random.uniform(0,1)>0.5: # 50% chance add white noise 0.005 - 0.01 rms
            bg_noise_waveform_cropped = transform_waveform(bg_noise_waveform_cropped,
                add_white_noise=random.uniform(0.005, 0.03)
            )
        if random.uniform(0,1)>0.5: # 50% chance add pink noise 0.005 - 0.01 rms
            bg_noise_waveform_cropped = transform_waveform(bg_noise_waveform_cropped,
                add_pink_noise=random.uniform(0.005, 0.03)
            )
        if random.uniform(0,1)>0.5: # 50% chance add brown noise 0.005 - 0.01 rms
            bg_noise_waveform_cropped = transform_waveform(bg_noise_waveform_cropped,
                add_brown_noise=random.uniform(0.005, 0.03)
            )
        # bg_noise_waveform_cropped = transform_waveform(bg_noise_waveform_cropped,
        #     add_pink_noise=0.005
        # )
        # set db
        bg_noise_waveform_cropped = transform_waveform(bg_noise_waveform_cropped, set_db=noise_db)

        # label += '_' + background_datatags[os.path.basename(bg_noise_path)[:-4]]['overlay_label']

        # highpass filter set by background noise tags data
        highpass_hz = background_datatags[os.path.basename(bg_noise_path)[:-4]].get('highpass', None)
        if highpass_hz:
            highpass_hz = int(highpass_hz)
        else:
            highpass_hz = 0
        highpass_hz += random.randint(0,config['output']['highpass_variable'])
        lowpass_hz = background_datatags[os.path.basename(bg_noise_path)[:-4]].get('lowpass', None)
        if lowpass_hz:
            lowpass_hz = int(lowpass_hz)
        else:
            lowpass_hz = (min(original_sample_rate,sample_rate)) / 2
        lowpass_hz -= random.randint(0,config['output']['lowpass_variable'])
        if specify_bandpass is not None:
            highpass_hz, lowpass_hz = specify_bandpass

        # adding random number of negative noises (cars, rain, wind). 
        # no boxes stored for these, as they are treated like background noise
        n_negative_overlays = random.randint(negative_overlay_range[0], negative_overlay_range[1])
        for j in range(n_negative_overlays):
            negative_segment_path = random.choice(negative_segment_paths)
            # label += '_'+negative_datatags[os.path.basename(negative_segment_path)[:-4]]['overlay_label']
            
            negative_waveform, neg_sr = load_waveform(negative_segment_path)

            neg_db = 10*torch.log10(torch.tensor(random.uniform(snr_range[0], snr_range[1])))+noise_db
            negative_waveform = transform_waveform(negative_waveform, resample=[neg_sr,sample_rate], set_db=neg_db)
            
            negative_waveform_cropped, start = crop_overlay_waveform(bg_noise_waveform_cropped.shape[1], negative_waveform)

            overlay = torch.zeros_like(bg_noise_waveform_cropped)
            overlay[:,max(0,start) : max(0,start) + negative_waveform_cropped.shape[1]] = negative_waveform_cropped
            bg_noise_waveform_cropped += overlay
            # label += 'p' + f"{(10 ** ((neg_db - noise_db) / 10)).item():.3f}" # power label
            
        new_waveform = bg_noise_waveform_cropped.clone()
        bg_spec_temp = transform_waveform(bg_noise_waveform_cropped, to_spec='power')
        bg_time_bins, bg_freq_bins = bg_spec_temp.shape[2], bg_spec_temp.shape[1]
        freq_bins_cutoff_bottom = int((highpass_hz / (sample_rate / 2)) * bg_freq_bins)
        freq_bins_cutoff_top = int((lowpass_hz / (sample_rate / 2)) * bg_freq_bins)

        # Adding random number of positive vocalisation noises
        # initialise label arrays
        boxes = []
        classes = []
        n_positive_overlays = random.randint(positive_overlay_range[0], positive_overlay_range[1])
        print(f'\n{idx}:    creating new image with {n_positive_overlays} positive overlays, bg={os.path.basename(bg_noise_path)}')
        succuessful_positive_overlays = 0
        while_catch = 0
        while succuessful_positive_overlays < n_positive_overlays:
            while_catch += 1
            if while_catch > 100:
                print(f"{idx}: Error, too many iterations")
                break

            # select positive overlay
            if specify_positive is not None:
                positive_segment_path = specify_positive
            else: 
                positive_segment_path = random.choice(positive_segment_paths)
                    
            # check if 'species' is 'chorus' (regardless of single_class because this determines how things are placed in the 10s image)
            # if positive_datatags[os.path.basename(positive_segment_path)[:-4]]['species'] == 'chorus':
                # continue # #TODO fix skip chorus
                # if classes.count(1) > 0:
                    # continue # only one chorus per image
                # species_class=1
            # else:
                # species_class=0
            species_class = positive_datatags[os.path.basename(positive_segment_path)[:-4]].get('species', None)
            if not species_class:
                species_class = 'unknown'

            positive_waveform, pos_sr = load_waveform(positive_segment_path)
            positive_waveform = transform_waveform(positive_waveform, resample=[pos_sr,sample_rate])
            positive_waveform_cropped, start = crop_overlay_waveform(bg_noise_waveform_cropped.shape[1], positive_waveform)

            pos_db_normalising = -9
            positive_waveform_cropped = transform_waveform(positive_waveform_cropped, set_db=pos_db_normalising)
            
            # attempt to place segment at least 1 seconds from other starts #TODO this introduces a bias
            # if positive_waveform.shape[1] < bg_noise_waveform_cropped.shape[1]:
            #     for i in range(20):
            #         positive_waveform_cropped, start = crop_overlay_waveform(bg_noise_waveform_cropped.shape[1], positive_waveform)
                    # if not any([start < box[0] + 1*sample_rate and start > box[0] - 1*sample_rate for box in boxes]):
                    #     break

            threshold = 2 # PSNR, db
            band_check_width = 5 # 5 bins
            edge_avoidance = 0.005 # 0.5% of final image per side, 50 milliseconds 120 Hz rounds to 4 and 5 bins -> 43 milliseconds 117 Hz
            freq_edge, time_edge = int(edge_avoidance*bg_freq_bins), int(edge_avoidance*bg_time_bins)
            # first pass find frequency top and bottom
            positive_spec_temp = transform_waveform(positive_waveform_cropped, to_spec='power')
            seg_freq_bins, seg_time_bins = positive_spec_temp.shape[1], positive_spec_temp.shape[2]
            start_time_bins = int(start * bg_time_bins / bg_noise_waveform_cropped.shape[1])
            first_pass_freq_start, first_pass_freq_end=None, None
            for i in range(max(freq_edge,freq_bins_cutoff_bottom), min(seg_freq_bins-freq_edge,freq_bins_cutoff_top)-1-band_check_width):
                PS_avg = torch.mean(torch.tensor([positive_spec_temp[:,j:j+1,:].max() for j in range(i,i+band_check_width)]))
                N_avg = torch.mean(torch.tensor([
                    bg_spec_temp[:,
                        j:j+1,
                        max(start_time_bins,time_edge):min(start_time_bins+seg_time_bins,bg_time_bins-time_edge)
                    ].mean() for j in range(i,i+band_check_width)]
                ))
                if (10*torch.log10(PS_avg / N_avg) > threshold) and (PS_avg > threshold):
                    first_pass_freq_start = i
                    break
            for i in range(min(seg_freq_bins-freq_edge, freq_bins_cutoff_top)-1, max(freq_edge,freq_bins_cutoff_bottom)+band_check_width, -1):
                PS_avg = torch.mean(torch.tensor([positive_spec_temp[:,j:j+1,:].max() for j in range(i-band_check_width,i)]))
                N_avg = torch.mean(torch.tensor([
                    bg_spec_temp[:,
                        j:j+1,
                        max(start_time_bins,time_edge):min(start_time_bins+seg_time_bins,bg_time_bins-time_edge)
                    ].mean() for j in range(i-band_check_width,i)]
                ))
                if (10*torch.log10(PS_avg / N_avg) > threshold) and (PS_avg > threshold):
                    first_pass_freq_end = i
                    break
            if (first_pass_freq_start and first_pass_freq_end) and (first_pass_freq_end > first_pass_freq_start) and (start_time_bins+seg_time_bins < bg_time_bins):
                #calculate noise power at box
                full_spec = torch.zeros_like(bg_spec_temp[:, :, max(0,start_time_bins):start_time_bins+seg_time_bins])
                full_spec[:, first_pass_freq_start:first_pass_freq_end, :] = bg_spec_temp[:, first_pass_freq_start:first_pass_freq_end, max(0,start_time_bins):start_time_bins+seg_time_bins]
                waveform_at_box = torchaudio.transforms.GriffinLim(
                    n_fft=2048, 
                    win_length=2048, 
                    hop_length=512, 
                    power=2.0
                )(full_spec)
                noise_db_at_box = 10*torch.log10(torch.mean(torch.square(waveform_at_box)))

                pos_snr = torch.tensor(random.uniform(snr_range[0], snr_range[1]))
                pos_db = 10*torch.log10(pos_snr)+noise_db_at_box
                # power shift signal
                positive_waveform_cropped = transform_waveform(positive_waveform_cropped, set_db=pos_db)
                # dynamically find the new bounding box after power shift
                pos_spec_temp = transform_waveform(positive_waveform_cropped, to_spec='power')
            else:
                print(f"{idx}: Error, unable to find bounding box for {positive_datatags[os.path.basename(positive_segment_path)[:-4]]['overlay_label']}")
                # which was the error
                print(f"first_pass_freq_start: {first_pass_freq_start}, first_pass_freq_end: {first_pass_freq_end}")
                print(f"end_time_bins: {start_time_bins+seg_time_bins}, bg_time_bins: {bg_time_bins}")
                print(f"seg_freq_bins: {seg_freq_bins}, seg_time_bins: {seg_time_bins}")
                continue

            found=0
            # if seg_time_bins < bg_time_bins:
            if True:
                # Find frequency edges (vertical scan)
                freq_start = max(freq_edge,freq_bins_cutoff_bottom) # from the bottom up
                for i in range(max(freq_edge,freq_bins_cutoff_bottom), min(seg_freq_bins-freq_edge,freq_bins_cutoff_top)-1-band_check_width):
                    N_avg = torch.mean(torch.tensor([
                        bg_spec_temp[:,
                            j:j+1,
                            max(start_time_bins,time_edge):min(start_time_bins+seg_time_bins,bg_time_bins-time_edge)
                        ].mean() for j in range(i,i+band_check_width)]
                    ))
                    PS_avg = torch.mean(torch.tensor([pos_spec_temp[:,j:j+1,:].max() for j in range(i,i+band_check_width)]))
                    if (10*torch.log10(PS_avg / N_avg) > threshold) and (PS_avg > threshold):
                        freq_start = i
                        found+=1
                        break

                freq_end = min(seg_freq_bins-freq_edge, freq_bins_cutoff_top)-1 # from the top down
                for i in range(min(seg_freq_bins-freq_edge, freq_bins_cutoff_top)-1, max(freq_edge,freq_bins_cutoff_bottom)+band_check_width, -1):
                    N_avg = torch.mean(torch.tensor([
                        bg_spec_temp[:,
                            j:j+1,
                            max(start_time_bins,time_edge):min(start_time_bins+seg_time_bins,bg_time_bins-time_edge)
                        ].mean() for j in range(i-band_check_width,i)]
                    ))
                    PS_avg = torch.mean(torch.tensor([pos_spec_temp[:,j:j+1,:].max() for j in range(i-band_check_width,i)]))
                    if (10*torch.log10(PS_avg / N_avg) > threshold) and (PS_avg > threshold):
                        freq_end = i
                        found+=1
                        break

                # Find time edges (horizontal scan)
                start_time_offset = 0 # from the left
                if freq_start < freq_end:
                    for i in range(0, seg_time_bins-1-band_check_width):
                        N_avg = torch.mean(torch.tensor([
                            bg_spec_temp[:,
                                freq_start:freq_end,
                                j:j+1
                            ].mean() for j in range(i,i+band_check_width)]
                        ))
                        PS_avg = torch.mean(torch.tensor([pos_spec_temp[:,freq_start:freq_end,j:j+1].max() for j in range(i,i+band_check_width)]))
                        if (10*torch.log10(PS_avg / N_avg) > threshold) and (PS_avg > threshold):
                            start_time_offset = i
                            found+=1
                            break

                    end_time_offset = seg_time_bins - 1 # from the right
                    for i in range(seg_time_bins - 1, 0+band_check_width, -1):
                        N_avg = torch.mean(torch.tensor([
                            bg_spec_temp[:,
                                freq_start:freq_end,
                                j:j+1
                            ].mean() for j in range(i-band_check_width,i)]
                        ))
                        PS_avg = torch.mean(torch.tensor([pos_spec_temp[:,freq_start:freq_end,j:j+1].max() for j in range(i-band_check_width,i)]))
                        if (10*torch.log10(PS_avg / N_avg) > threshold) and (PS_avg > threshold):
                            end_time_offset = i
                            found+=1
                            break

            # TODO maybe remove?: noises longer than final length are treated as continuous, no need for time edges
            #TODO: tripple check iou ios merging calcualtions due to format change
            elif seg_time_bins >= bg_time_bins:
                # Find frequency edges (vertical scan) - minimum start at 2 (~100 Hz @ 48khz) to avoid low frequency interferance
                freq_start = freq_edge
                for i in range(max(freq_edge,freq_bins_cutoff_bottom), min(seg_freq_bins-freq_edge,freq_bins_cutoff_top)-1):
                    N = bg_spec_temp[:,
                        i:i+1,time_edge:bg_time_bins-time_edge
                    ].mean()
                    PS = pos_spec_temp[:,i:i+1,time_edge:seg_time_bins-time_edge].max()
                    if (10*torch.log10(PS / N) > threshold) and (PS > threshold):
                        freq_start = i
                        found+=1
                        break
                freq_end = seg_freq_bins - 1
                for i in range(min(seg_freq_bins, freq_bins_cutoff_top)-1, max(2,freq_bins_cutoff_bottom), -1):
                    N = bg_spec_temp[:,
                        i:i+1,time_edge:bg_time_bins-time_edge
                    ].mean()
                    PS = pos_spec_temp[:,i:i+1,time_edge:seg_time_bins-time_edge].max()
                    if (10*torch.log10(PS / N) > threshold) and (PS > threshold):
                        freq_end = i
                        found+=1
                        break
                if freq_start < freq_end:
                    start_time_offset = 0
                    end_time_offset = seg_time_bins - 1
                    found+=2

            # verify height and width are not less than 1% of the final image
            if ((freq_end - freq_start)/bg_freq_bins) < 0.0065 or ((end_time_offset - start_time_offset)/bg_time_bins) < 0.0065:
                print(f"{idx}: Error, too small, power {pos_db-noise_db:.3f}, freq {(freq_end - freq_start)/bg_freq_bins:.3f}, time {(end_time_offset - start_time_offset)/bg_time_bins:.3f}")
                continue
            if ((freq_end - freq_start)/bg_freq_bins) > 0.99 or found < 4:
                print(f"{idx}: Error, too faint, power {pos_db-noise_db:.3f}")
                continue

            # ## Paper small square Plot
            # combined_for_plot = bg_noise_waveform_cropped.clone()
            # combined_for_plot[:,max(0,start) : max(0,start) + positive_waveform_cropped.shape[1]] += positive_waveform_cropped
            # temp_comobined_spec = transform_waveform(combined_for_plot, to_spec='power')
            # plot_spectrogram(paths=['x'], not_paths_specs=[temp_comobined_spec],
            #     logscale=False, 
            #     color='bw',
            #     draw_boxes=[[
            #         [10, seg_time_bins+10, first_pass_freq_start, first_pass_freq_end],
            #         [start_time_offset+10, end_time_offset+10, freq_start, freq_end]
            #         ]],
            #     box_format='xxyy',
            #     set_width=1,fontsize=18,
            #     box_colors=['#00eaff','#45ff45'],
            #     box_styles=['solid','--'],
            #     box_widths=[3,3],
            #     crop_time=[max(0,start_time_bins-15), min(start_time_bins+seg_time_bins+15,bg_time_bins)],
            #     crop_frequency=[max(first_pass_freq_start-15,0), min(first_pass_freq_end+15,bg_freq_bins)],
            #     specify_freq_range=[((first_pass_freq_start-15)/bg_freq_bins)*24000, ((first_pass_freq_end+15)/bg_freq_bins)*24000]
            # )

            def appendSpeciesClass(classes, species_class):
                # print(f' {species_class} ', end='')
                if species_class in species_value_map:
                    classes.append(species_value_map[species_class])
                else:
                    classes.append(len(species_value_map))
                    species_value_map[species_class] = len(species_value_map)
                    write_species_value_map_to_file(species_value_map, save_directory)
                # print(f'    {species_class}')
                return classes

            overlay = torch.zeros_like(bg_noise_waveform_cropped)
            overlay[:,max(0,start) : max(0,start) + positive_waveform_cropped.shape[1]] = positive_waveform_cropped
            new_waveform += overlay
            succuessful_positive_overlays += 1

            freq_start, freq_end = map_frequency_to_log_scale(bg_freq_bins, [freq_start, freq_end])
            # add bounding box to list, in units of spectrogram time and log frequency bins
            boxes.append([max(start_time_offset,start_time_bins+start_time_offset), max(end_time_offset, start_time_bins+end_time_offset), freq_start, freq_end])
            classes = appendSpeciesClass(classes, species_class)
            print(f'{idx}:    overlaying {os.path.basename(positive_segment_path)[:-4]} at {start_time_offset} - {end_time_offset}, freq {freq_start} - {freq_end}, power {pos_db-noise_db:.3f}, snr {pos_snr:.1f}')

            if config['output']['include_coco_masks']:
                # Generate mask annotation
                mask_annotation = generate_masks(
                    overlay_waveform=overlay, # This 'overlay' is the isolated positive signal on zeroed background
                    image_id=idx,
                    category_id=classes[-1],
                    last_box=boxes[-1], # This is the box of the overlay in full spectrogram coords
                    threshold_db=config['output'].get('mask_threshold_db', 10),
                    log_scale=config['output'].get('log_scale_masks', True),
                    debug=False
                )
                coco_annotations.append(mask_annotation)

            if config['output']['include_yolo_masks']:
                # 1. Generate the binary mask for this specific 'overlay'
                #    The 'overlay' tensor here is the waveform of the single positive event placed on a zero background.
                overlay_spec_for_mask = transform_waveform(overlay, to_spec='power')
                if config['output'].get('log_scale_masks', True): # Assumes log_scale for masks
                    overlay_spec_for_mask = log_scale_spectrogram(overlay_spec_for_mask)
                
                spec_db_for_mask = 10 * torch.log10(overlay_spec_for_mask + 1e-10)
                binary_mask_overlay = (spec_db_for_mask > config['output'].get('mask_threshold_db', 10)).numpy().astype(np.uint8)
                binary_mask_overlay = binary_mask_overlay[0] # remove batch dim

                # 2. Get the bounding box of this overlay in the full spectrogram coordinates
                #    boxes[-1] is [t_min_full_spec, t_max_full_spec, f_min_log_full_spec, f_max_log_full_spec]
                current_box_in_full_spec = boxes[-1]

                # 3. Get full spectrogram dimensions (time bins, log-scaled frequency bins)
                #    bg_time_bins is total time bins of the full spectrogram.
                #    bg_freq_bins is total frequency bins (used as the height for log-scaled spectrogram).
                full_spec_dims_yolo = (bg_time_bins, bg_freq_bins) 

                # 4. Call generate_yolo_segment_data_from_binary_mask
                yolo_strings_for_this_overlay = generate_yolo_segment_data_from_binary_mask(
                    binary_mask=binary_mask_overlay,
                    class_id=classes[-1], # current class ID
                    box_in_full_spec_bins=current_box_in_full_spec,
                    full_spec_dims_bins=full_spec_dims_yolo,
                    simplify_tolerance=config['output'].get('yolo_simplify_tolerance', 1.5)
                )

                # 5. Store these strings. `save_files_path` will be defined later in the loop
                #    So, we need to collect them temporarily per `idx` and then assign to `save_files_path` key.
                if idx not in yolo_data_for_files: # Using idx as a temporary key
                    yolo_data_for_files[idx] = []
                yolo_data_for_files[idx].extend(yolo_strings_for_this_overlay)

            # potentially repeat song
            if repetitions:
                if random.uniform(0,1)>0.5:
                    seg_samples = positive_waveform_cropped.shape[1]
                    separation = random.uniform(0.5, 2) # 0.5-3 seconds
                    separation_samples = int(separation*sample_rate)
                    n_repetitions = random.randint(repetitions[0], repetitions[1])
                    print(f'{idx}:    repeating {n_repetitions} times, separation {separation:.2f}s')
                    new_start = start
                    for i in range(n_repetitions):
                        new_start += seg_samples + separation_samples
                        if new_start + seg_samples < (bg_noise_waveform_cropped.shape[1]-1) and (new_start>0):
                            new_start_bins = int(new_start * bg_time_bins / bg_noise_waveform_cropped.shape[1])
                            overlay = torch.zeros_like(bg_noise_waveform_cropped)
                            overlay[:,new_start : new_start + positive_waveform_cropped.shape[1]] = positive_waveform_cropped
                            new_waveform += overlay
                            succuessful_positive_overlays += 1

                            boxes.append([new_start_bins+start_time_offset, new_start_bins+end_time_offset, freq_start, freq_end])
                            classes = appendSpeciesClass(classes, species_class)
                            # label += 'x' # repetition
                            if config['output']['include_coco_masks']:
                                mask_annotation = generate_masks(
                                    overlay_waveform=overlay,
                                    image_id=idx,
                                    category_id=classes[-1],
                                    last_box=boxes[-1],
                                    threshold_db=config['output'].get('mask_threshold_db', 10),
                                    log_scale=config['output'].get('log_scale_masks', True),
                                    debug=False
                                )
                                coco_annotations.append(mask_annotation)

                            if config['output']['include_yolo_masks']:
                                overlay_spec_for_mask_rep = transform_waveform(overlay, to_spec='power')
                                if config['output'].get('log_scale_masks', True):
                                    overlay_spec_for_mask_rep = log_scale_spectrogram(overlay_spec_for_mask_rep)
                                
                                spec_db_for_mask_rep = 10 * torch.log10(overlay_spec_for_mask_rep + 1e-10)
                                binary_mask_overlay_rep = (spec_db_for_mask_rep > config['output'].get('mask_threshold_db', 10)).numpy().astype(np.uint8)
                                binary_mask_overlay_rep = binary_mask_overlay_rep[0]

                                current_box_in_full_spec_rep = boxes[-1]
                                full_spec_dims_yolo_rep = (bg_time_bins, bg_freq_bins)

                                yolo_strings_for_this_overlay_rep = generate_yolo_segment_data_from_binary_mask(
                                    binary_mask=binary_mask_overlay_rep,
                                    class_id=classes[-1],
                                    box_in_full_spec_bins=current_box_in_full_spec_rep,
                                    full_spec_dims_bins=full_spec_dims_yolo_rep,
                                    simplify_tolerance=config['output'].get('yolo_simplify_tolerance', 0.1)
                                )
                                
                                # Using idx as a temporary key, same as above
                                if idx not in yolo_data_for_files:
                                    yolo_data_for_files[idx] = []
                                yolo_data_for_files[idx].extend(yolo_strings_for_this_overlay_rep)
                        else:
                            break
            
        final_audio = transform_waveform(new_waveform, to_spec='power')
        final_audio = spectrogram_transformed(
            final_audio,
            highpass_hz=highpass_hz,
            lowpass_hz=lowpass_hz
        )

        # final normalisation, which is applied to real audio also
        final_audio = spectrogram_transformed(
            final_audio,
            set_db=-10,
        )
        if not config['paths']['do_train_val_split']:
            save_files_path = f"{idx}"
        elif idx_offset > val_index:
            save_files_path = f"val/{idx}"
        else:
            save_files_path = f"train/{idx}"

        if config['output']['include_kaytoo']:
            # Append to the file within the loop
            wav_path = f"classifiers/augmented_dataset/sound_files/{save_files_path}"
            if not os.path.exists(os.path.dirname(wav_path)):
                os.makedirs(os.path.dirname(wav_path))
            spec_to_audio(final_audio, save_to=wav_path, energy_type='power')
            
            coarse_labels_output_path = f'classifiers/augmented_dataset/labels.csv'
            with open(coarse_labels_output_path, 'a') as f:
                if config['output']['single_class']:
                    if len(classes) > 0:
                        coarse_class = 1
                    else:
                        coarse_class = 0
                else:
                    if len(classes) > 0:
                        coarse_class = max(classes) + 1
                f.write(f'{idx}.wav,{coarse_class}\n')
        
        if config['output']['include_spectrogram']:
            image = spectrogram_transformed(
                final_audio,
                to_pil=True,
                color_mode=color_mode,
                log_scale=True,
                normalise='power_to_PCEN',
                resize=(640, 640),
            )
            image_output_path = f'{save_directory}/artificial_dataset/images/{save_files_path}.jpg'
        
            # check directory exists
            if not os.path.exists(os.path.dirname(image_output_path)):
                os.makedirs(os.path.dirname(image_output_path))
            image.save(image_output_path, format='JPEG', quality=95)
            # Reopen the image to check for errors (slow)
            # try:
            #     img = Image.open(image_output_path)
            #     img.load()  # loading of image data
            #     img.close()
            # except (IOError, SyntaxError) as e:
            #     print(f"Invalid image after reopening: {e}")

        if config['output']['include_boxes']:
            box_label_output_path = f'{save_directory}/artificial_dataset/box_labels/{save_files_path}.txt'

            # Merge boxes based on IoU
            merged_boxes, merged_classes = merge_boxes_by_class(boxes, classes, iou_threshold=0.1, ios_threshold=0.4)
            
            # use this to remember how to turn off log later
            # temp_unlog_boxes = []
            # for box in boxes:
            #     y1, y2 = map_frequency_to_linear_scale(bg_freq_bins, [box[2], box[3]])
            #     temp_unlog_boxes.append([box[0], box[1], y1, y2])
            # plot_spectrogram(
            #     paths=['x'],
            #     not_paths_specs=[final_audio],
            #     logscale=True,fontsize=16,set_width=1.5,
            #     draw_boxes=[temp_unlog_boxes],
            #     box_colors=['#45ff45']*len(boxes),
            #     box_widths=[2]*len(boxes),
            #     box_format='xxyy')
            # temp_unlog_boxes = []
            # for box in merged_boxes:
            #     y1, y2 = map_frequency_to_linear_scale(bg_freq_bins, [box[2], box[3]])
            #     temp_unlog_boxes.append([box[0], box[1], y1, y2])
            # temp_pcen_spec = pcen(final_audio)
            # plot_spectrogram(
            #     paths=['x'],
            #     not_paths_specs=[temp_pcen_spec],color_mode='HSV',to_db=False,
            #     logscale=True,fontsize=15,set_width=1.3,
            #     draw_boxes=[temp_unlog_boxes],
            #     box_colors=['white']*len(merged_boxes),
            #     box_widths=[2]*len(merged_boxes),
            #     box_format='xxyy')
            
            # make label txt file
            # check directory exists
            if not os.path.exists(os.path.dirname(box_label_output_path)):
                os.makedirs(os.path.dirname(box_label_output_path))
            with open(box_label_output_path, 'w') as f:
                for box, species_class in zip(merged_boxes, merged_classes):
                    x_center = (box[0] + box[1]) / 2 / bg_time_bins
                    width = (box[1] - box[0]) / bg_time_bins

                    y_center = (box[2] + box[3]) / 2 / bg_freq_bins
                    y_center = 1 - y_center # vertical flipping for yolo
                    height = (box[3] - box[2]) / bg_freq_bins

                    if x_center < 0 or x_center > 1 or y_center < 0 or y_center > 1 or width < 0 or width > 1 or height < 0 or height > 1:
                        print(f"{idx}: Error, box out of bounds!\n\n******\n\n******\n\n*******\n\n")

                    # Write to file in the format [class_id x_center y_center width height]
                    f.write(f'{species_class} {x_center} {y_center} {width} {height}\n')
        
    # After the main loop, save all collected annotations
    if config['output']['include_yolo_masks'] and yolo_data_for_files: # Check if dict is not empty
        print(f"Preparing to save YOLO labels for {len(yolo_data_for_files)} images.")
        # yolo_data_for_files is currently keyed by `idx` (integer)
        
        for image_idx_key, segments in yolo_data_for_files.items():
            if not segments:
                print(f"No YOLO segments for image index {image_idx_key}, skipping.")
                continue

            # Reconstruct save_files_path based on image_idx_key (which is `idx`)
            # This logic must mirror how save_files_path was constructed inside the loop
            # Need start_idx and val_index from the main loop context
            current_idx_offset_for_path = image_idx_key - start_idx # Get the 0-based offset for this image
            
            yolo_label_file_base_path = ""
            if not config['paths']['do_train_val_split']:
                yolo_label_file_base_path = f"{image_idx_key}"
            elif current_idx_offset_for_path > val_index: # val_index was calculated based on n & val_ratio
                yolo_label_file_base_path = f"val/{image_idx_key}"
            else:
                yolo_label_file_base_path = f"train/{image_idx_key}"

            yolo_label_output_path = f'{save_directory}/artificial_dataset/yolo_labels/{yolo_label_file_base_path}.txt'
            yolo_label_dir = os.path.dirname(yolo_label_output_path)
            if not os.path.exists(yolo_label_dir):
                os.makedirs(yolo_label_dir, exist_ok=True)
            
            with open(yolo_label_output_path, 'w') as f_yolo_txt:
                for segment_line in segments:
                    f_yolo_txt.write(segment_line + '\n')
            # print(f"Saved YOLO labels to {yolo_label_output_path}") # Too verbose for many files

        print(f"Finished saving YOLO label files.")

        # Create dataset.yaml
        dataset_yaml_path = f'{save_directory}/artificial_dataset/dataset.yaml'
        # Invert species_value_map: {index: name} -> {str(index): name}
        names_map = {str(v): k for k, v in species_value_map.items()} 
        
        # Determine 'path' for dataset.yaml correctly
        # dataset.yaml is in save_directory/artificial_dataset/
        # image paths are relative to artificial_dataset/
        # So, path: ../ (relative to artificial_dataset dir, if default_datasets_dir is one level up)
        # or path: . (if paths in train/val are relative to artificial_dataset/)
        # Ultralytics usually expects 'path' to be the root of the dataset,
        # and train/val to be relative to that.
        # If dataset.yaml is in 'artificial_dataset', and images are in 'artificial_dataset/images/train',
        # then path: . and train: images/train is correct.

        yaml_data = {
            'path': '.', # Root directory relative to this YAML file.
            'nc': len(names_map),
            'names': names_map
        }
        if config['paths']['do_train_val_split']:
            yaml_data['train'] = 'images/train' # relative to 'path'
            yaml_data['val'] = 'images/val'   # relative to 'path'
            # test: can be added if a test set is created: 'images/test'
        else:
            yaml_data['train'] = 'images' # All images in 'images' directory
            # yaml_data['val'] = 'images' # Or omit if no dedicated val set

        with open(dataset_yaml_path, 'w') as f_yaml:
            yaml.dump(yaml_data, f_yaml, sort_keys=False, default_flow_style=None)
        print(f"Saved YOLO dataset YAML to {dataset_yaml_path}")

    elif config['output']['include_coco_masks']:
        # Finalize and save COCO dataset (if it was populated)
        if coco_dataset and coco_annotations: # Ensure it was initialized and has data
            # Add images info (ensure this covers all images, including those in val split)
            # This needs to be done carefully if train/val split is active.
            # The original COCO image population was `for i in range(n): coco_dataset['images'].append(...)`
            # This assumes image filenames are just "i.jpg". With train/val, they are "train/i.jpg".
            
            # Clear and repopulate images based on actual saved files if necessary, or ensure filenames are consistent.
            # For now, let's assume the original COCO image population logic is sufficient if filenames are just "idx.jpg".
            # If `save_files_path` is "train/idx.jpg", then coco_dataset['images'] file_name needs to reflect that.
            
            # Let's adjust COCO image population to be more robust with train/val splits
            coco_dataset['images'] = [] # Clear any previous/simple population
            for i_img in range(n): # Iterate up to total number of images
                idx_val_check = start_idx + i_img # The actual index used for train/val decision
                img_file_name_coco = ""
                if not config['paths']['do_train_val_split']:
                    img_file_name_coco = f"{idx_val_check}.jpg"
                elif idx_val_check > val_index: # val_index is calculated based on n and val_ratio
                    img_file_name_coco = f"val/{idx_val_check}.jpg"
                else:
                    img_file_name_coco = f"train/{idx_val_check}.jpg"
                
                coco_dataset['images'].append({
                    'id': idx_val_check, # Use the actual image index as ID
                    'file_name': img_file_name_coco,
                    'width': config['output'].get('image_width', 640), # Use configured or default
                    'height': config['output'].get('image_height', 640)
                })

            coco_dataset['categories'] = [] # Clear and repopulate
            for species, cat_idx in species_value_map.items():
                coco_dataset['categories'].append({
                    'id': cat_idx,
                    'name': species,
                    'supercategory': 'vocalisation' # Or make configurable
                })
            
            coco_dataset['annotations'] = coco_annotations # These were collected with correct image_id (idx)
            
            coco_output_path = f'{save_directory}/artificial_dataset/mask_annotations.json'
            with open(coco_output_path, 'w') as f_coco_json:
                json.dump(coco_dataset, f_coco_json, indent=2) # Add indent for readability
            print(f"Saved COCO annotations to {coco_output_path}")

    if plot:
        plot_labels(config, [0,n], save_directory)

def run_augmentation(config=None):
    """Main function to run the augmentation pipeline with given config"""
    if config is None:
        # Load default config if none provided
        with open('config.yaml') as f:
            config = yaml.safe_load(f)
    
    dataset_path = config['paths']['dataset']
    background_path = config['paths']['noise']
    positive_paths = config['paths']['vocalisations']
    negative_paths = config['paths']['negative']

    if config['output']['rainbow_frequency']:
        color_mode = 'HSV'
    else:
        color_mode = 'BW'

    print(f'Generating overlays for {config["output"]["n"]} images')

    # generate overlays
    generate_overlays(
        config,
        get_data_paths = [dataset_path, background_path, positive_paths, negative_paths],
        save_directory = config['paths']['output'],
        n=config['output']['n'],
        clear_dataset=config['output']['overwrite_output_path'],
        sample_rate=48000,
        final_length_seconds=config['output']['length'],
        positive_overlay_range=config['output']['positive_overlay_range'],
        negative_overlay_range=config['output']['negative_overlay_range'],
        val_ratio=config['output']['val_ratio'],
        snr_range=config['output']['snr_range'],
        save_wav=config['output']['include_soundfile'],
        plot=config['plot']['toggle'],
        color_mode=color_mode,
        repetitions=config['output']['repetitions'],
        specify_noise=None
    )

if __name__ == "__main__":
    run_augmentation()