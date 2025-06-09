# this script is used to visualise the results of the classifiers (inference results as a density map). it also calculates and adds files for the continuous segments

import os
from pathlib import Path
import pandas as pd
import numpy as np
import wave
import contextlib
import matplotlib.pyplot as plt

# Directory to site name mapping
mapping = {
    "1d - Mona Vale": "1d - Mona Vale",
    "2a - Ilam Grdns": "2a - Ilam Gardens",
    "2d - Fndltn Park": "2d - Fendalton park ",
    "2e- Ricc Bush": "2e - Riccarton Bush",
    "2f - Ilam fields": "2f - Ilam fields",  # Not in CSV, kept as is
    "3a- Bttle Lake": "3a - Bottle lake",
    "3b- Spencer Prk": "3b - Spencer park",
    "3e- Arden Res": "3e - Arden reserve",
    "4a - Groynes": "4a - The Groynes",
    "4b - Styx M": "4b - Styx Mill",
    "4c - Barnes Res": "4c - Barnes reserve",
    "4e - Springvale G": "4e - Springvale garden",
    "5a - Vic park": "5a - Victoria park",
    "5d - Hansen Park": "5d - Hansens park",
    "5e - Mt Vernon Track": "5e - Mt Vernon track",
    "8a - Cant Agri": "8a - Canterbury agricultural park",
    "8b - Seager": "8b - Seager Park",
    "8d - Glynne": "8d - Glyne Reserve",
    "8e - Warren": "8e - Warren park",
    "9a - NB Redzone": "9a - NB Redzone",
    "9b - Avonside Redzone": "9b -  Avonside Redzone",
    "AudioMoth test": "AudioMoth test",  # Not in CSV, kept as is
    "Motukarara": "Motukarara"  # Not in CSV, kept as is
}

inference_results_dir = 'classifiers/inference_results/consensus_Exp_2443_e5_Exp_2462_e5'
audio_base_dir = '/Volumes/Rectangle/little_owl/data-backup/Little Owl AudioMoth'
corrupted_files_path = 'corrupted_files.csv' #filename
high_positive_results = 'high_positive_results.csv' #filename,probability,start_time_seconds,start_time
positive_results = 'positive_results.csv' #filename,probability,start_time_seconds,start_time
coordinates = Path('/Volumes/Rectangle/little_owl/data-backup/Little Owl AudioMoth/survey_sites_coordinates.csv') #name, latitude, longitude (decimal degrees)
coord_df = pd.read_csv(coordinates)

# List to keep track of directories with no coordinates
no_coordinates = ["2f - Ilam fields", "AudioMoth test", "Motukarara"]  # Known directories without coordinates

# Function to convert a CSV of positive detections to continuous segments for duration stats
def convert_to_continuous_segments(positive_csv, save_path=None):
    # Sort by filename and start_time_seconds
    positive_csv = positive_csv.sort_values(by=['filename', 'start_time_seconds'])
    
    continuous_segments = []
    current_segment = None
    
    for _, row in positive_csv.iterrows():
        if current_segment is None:
            # Start a new segment
            current_segment = {
                'filename': row['filename'],
                'start_time_seconds': row['start_time_seconds'],
                'start_time': row['start_time'],
                'end_time_seconds': row['start_time_seconds'] + 10
            }
        else:
            # Check if this detection should be part of the current segment
            time_diff = row['start_time_seconds'] - current_segment['start_time_seconds']
            if (row['filename'] == current_segment['filename'] and 
                time_diff <= 10):  # Less than or equal to 10 seconds apart
                # Extend the current segment
                current_segment['end_time_seconds'] = row['start_time_seconds'] + 10
            else:
                # Save the current segment and start a new one
                duration = current_segment['end_time_seconds'] - current_segment['start_time_seconds']
                if duration > 10:
                    continuous_segments.append({
                        'filename': current_segment['filename'],
                        'start_time_seconds': current_segment['start_time_seconds'],
                        'start_time': current_segment['start_time'],
                        'duration': duration
                    })
                current_segment = {
                    'filename': row['filename'],
                    'start_time_seconds': row['start_time_seconds'],
                    'start_time': row['start_time'],
                    'end_time_seconds': row['start_time_seconds'] + 10
                }
    
    # Don't forget to add the last segment
    if current_segment is not None:
        duration = current_segment['end_time_seconds'] - current_segment['start_time_seconds']
        if duration > 10:
            continuous_segments.append({
                'filename': current_segment['filename'],
                'start_time_seconds': current_segment['start_time_seconds'],
                'start_time': current_segment['start_time'],
                'duration': duration
            })
    
    # Convert to DataFrame
    result_df = pd.DataFrame(continuous_segments)
    
    # Save if path provided
    if save_path and result_df.shape[0] > 0:
        result_df.to_csv(save_path, index=False)
    
    return result_df
    
# Function to get duration of a WAV file
def get_wav_duration(wav_file):
    try:
        with contextlib.closing(wave.open(wav_file, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
            return duration
    except:
        return 0  # Return 0 for corrupted files
    

points = []
for dir_path in os.listdir(inference_results_dir):
    if not os.path.isdir(os.path.join(inference_results_dir, dir_path)):
        continue
    print(f'Processing {dir_path}')
    
    # Check if directory name is in our mapping
    csv_name = mapping.get(dir_path, dir_path)  # Use mapped name or original if not in mapping
    
    # Check if the mapped directory name is in the coordinates file
    if csv_name in coord_df['Name'].values:
        point = {}
        # Get the latitude and longitude for this directory
        site_coords = coord_df[coord_df['Name'] == csv_name]
        point['latitude'] = site_coords['Latitude'].values[0]
        point['longitude'] = site_coords['Longitude'].values[0]
        point['name'] = dir_path  # Keep original dir name for display
        
        # Load the results for this directory
        p_c = os.path.join(inference_results_dir, dir_path, corrupted_files_path)
        p_h = os.path.join(inference_results_dir, dir_path, high_positive_results)
        p_p = os.path.join(inference_results_dir, dir_path, positive_results)
        
        corrupted_files, high_positive, positive = [], [], []
        if os.path.exists(p_c):
            corrupted_files = pd.read_csv(p_c)
            point['corrupted_files'] = corrupted_files
        if os.path.exists(p_h):
            high_positive = pd.read_csv(p_h)
            point['high_positive'] = high_positive
        if os.path.exists(p_p):
            positive = pd.read_csv(p_p)
            point['positive'] = positive
        # convert high positive detections to continuous segments if needed
        total_high_positive_duration = 0
        if not high_positive.empty:
            high_continuous_segments_save_path = os.path.join(inference_results_dir, dir_path, 'high_positive_continuous_segments.csv')
            # if it exists, dont convert again
            if not os.path.exists(high_continuous_segments_save_path):
                high_positive_segments = convert_to_continuous_segments(high_positive, save_path=high_continuous_segments_save_path)
            else:
                high_positive_segments = pd.read_csv(high_continuous_segments_save_path)
            if high_positive_segments.shape[0] > 0:
                    
                point['high_positive_segments'] = high_positive_segments
                total_high_positive_duration = high_positive_segments['duration'].sum()
                point['total_high_positive_duration_hours'] = total_high_positive_duration / 3600
            else:
                point['total_high_positive_duration_hours'] = 0
        else:
            point['total_high_positive_duration_hours'] = 0
        
        # Calculate total audio duration
        total_duration = 0
        valid_files = 0
        total_files = 0
        
        audio_dir = os.path.join(audio_base_dir, dir_path)
        if os.path.exists(audio_dir):
            for root, _, files in os.walk(audio_dir):
                for file in files:
                    if file.lower().endswith(('.wav')):
                        total_files += 1
                        file_path = os.path.join(root, file)
                        if file not in corrupted_files:
                            try:
                                duration = get_wav_duration(file_path)
                                if duration > 0:
                                    total_duration += duration
                                    valid_files += 1
                            except:
                                pass  # Skip files that can't be read
        
        point['total_duration_hours'] = total_duration / 3600
        point['valid_files'] = valid_files
        point['total_files'] = total_files
        point['all_corrupted'] = (valid_files == 0 and total_files > 0)
        if total_duration > 0:
            normalised_vocalisation_duration = total_high_positive_duration / total_duration
        else:
            normalised_vocalisation_duration = 0
        point['normalised_vocalisation_duration'] = normalised_vocalisation_duration
        
        points.append(point)
        print(f'Finished {dir_path} - Total duration: {total_duration/3600:.2f} hours, Valid files: {valid_files}/{total_files} ({normalised_vocalisation_duration:.2f}% high positive)')
    else:
        if dir_path not in no_coordinates:
            print(f'Warning: No coordinates found for {dir_path} (mapped to {csv_name})')
            no_coordinates.append(dir_path)

# Extract data for plotting
latitudes = []
longitudes = []
site_names = []
high_positive_duration = []
normalized_durations = []
is_all_corrupted = []
recording_durations = []  # Store durations for point sizing
for point in points:
    latitudes.append(point['latitude'])
    longitudes.append(point['longitude'])
    site_names.append(point['name'])
    recording_durations.append(point['total_duration_hours'])  # Get durations
    
    # Get high positive counts
    if 'total_high_positive_duration_hours' in point:
        duration = point['total_high_positive_duration_hours']
    else:
        duration = 0
    high_positive_duration.append(duration)
    
    # Calculate normalized counts (per hour of valid audio)
    normalized = point['normalised_vocalisation_duration']
    normalized_durations.append(normalized)
    
    # Track if site has only corrupted files
    is_all_corrupted.append(point['all_corrupted'])

# Scale durations for point sizes (min 50, max 500)
min_size = 50
max_size = 500
if recording_durations:
    max_duration = max(d for d in recording_durations if d > 0) or 1
    scaled_sizes = [min_size + (d/max_duration)*(max_size-min_size) for d in recording_durations]
else:
    scaled_sizes = [min_size] * len(latitudes)

# Create the scatter plot
plt.figure(figsize=(11, 9))

# Plot regular points (not all corrupted)
regular_indices = [i for i, corrupted in enumerate(is_all_corrupted) if not corrupted]
# remove points in no_coordinates
regular_indices = [i for i in regular_indices if site_names[i] not in no_coordinates]
if regular_indices:
    # Split into points with and without high positive detections
    pos_indices = [i for i in regular_indices if high_positive_duration[i] > 0]
    zero_indices = [i for i in regular_indices if high_positive_duration[i] == 0]
    
    # Plot points with positive detections (filled)
    if pos_indices:
        pos_lats = [latitudes[i] for i in pos_indices]
        pos_longs = [longitudes[i] for i in pos_indices]
        pos_counts = [normalized_durations[i] for i in pos_indices]
        pos_sizes = [scaled_sizes[i] for i in pos_indices]
        
        scatter = plt.scatter(pos_longs, pos_lats, c=pos_counts, 
                            cmap='viridis', s=pos_sizes, alpha=0.7, edgecolors='black')
    
    # Plot points with zero high positive (empty circles)
    if zero_indices:
        zero_lats = [latitudes[i] for i in zero_indices]
        zero_longs = [longitudes[i] for i in zero_indices]
        zero_sizes = [scaled_sizes[i] for i in zero_indices]
        
        plt.scatter(zero_longs, zero_lats, s=zero_sizes, 
                  facecolors='none', edgecolors='black', alpha=0.7)
    
    # Add colorbar (if we have positive points)
    if pos_indices:
        cbar = plt.colorbar(scatter)
        cbar.set_label('Normalised detection duration (size indicates total audio)', labelpad=20, fontsize=12)

# Plot points with only corrupted files as crosses
corrupted_indices = [i for i, corrupted in enumerate(is_all_corrupted) if corrupted]
if corrupted_indices:
    corr_lats = [latitudes[i] for i in corrupted_indices]
    corr_longs = [longitudes[i] for i in corrupted_indices]
    
    plt.scatter(corr_longs, corr_lats, marker='x', color='red', s=100, 
               label='Sites with only corrupted files')

# Add labels to each point
for i, name in enumerate(site_names):
    plt.annotate(name, (longitudes[i], latitudes[i]), 
                xytext=(8, 0), textcoords='offset points')

plt.xlabel('Longitude', fontsize=12)
plt.ylabel('Latitude', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()


# Create a simple report with site statistics
report_data = []
for point in points:
    report_data.append({
        'Site': point['name'],
        'Site Name': mapping.get(point['name'], point['name']),
        'Latitude': point['latitude'],
        'Longitude': point['longitude'],
        'Total Files': point['total_files'],
        'Valid Files': point['valid_files'],
        'All Corrupted': point['all_corrupted'],
        'Total Recordings (hours)': point['total_duration_hours'],
        'Positive Detections': len(point.get('positive', [])),
        'High Positive Detections': len(point.get('high_positive', [])),
        'High Positive Duration (hours)': point['total_high_positive_duration_hours'],
        'Normalised Duration': point['normalised_vocalisation_duration']
    })
# print global stats
print(f'Total audio (non-corrupted): {sum(recording_durations):.2f} hours')
print(f"Total non-corrupted locations: {len([point for point in points if not point['all_corrupted']])}")
print(f'Total high-positive duration: {sum(high_positive_duration):.2f} hours')
print(f"Total high-positive locations: {len([point for point in points if point['total_high_positive_duration_hours'] > 0])}")
print(f"Total high-positive locations with valid audio: {len([point for point in points if point['total_high_positive_duration_hours'] > 0 and point['total_duration_hours'] > 0])}")
print(f"Total high-positive locations with no valid audio: {len([point for point in points if point['total_high_positive_duration_hours'] > 0 and point['total_duration_hours'] == 0])}")
print(f'Average normalised vocalisation duration: {np.mean(normalized_durations):.2f}')
print(f'Total locations: {len(points)}')
print(f'Total files: {sum([point["total_files"] for point in points])}')
print(f'Total valid files: {sum([point["valid_files"] for point in points])}')
print(f'Total corrupted files: {sum([point["total_files"] - point["valid_files"] for point in points])}')
print(f'Total high positive detections: {sum([len(point.get("high_positive", [])) for point in points])}')
print(f'Total positive detections: {sum([len(point.get("positive", [])) for point in points])}')

report_df = pd.DataFrame(report_data)
report_df.to_csv('site_detection_statistics.csv', index=False)

plt.show()