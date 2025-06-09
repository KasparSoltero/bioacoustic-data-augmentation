import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os # Used for basic directory check initially

# --- Configuration ---
# Use the directory provided by the user
wav_dir = '/Volumes/Rectangle/little_owl/noise'

# --- Initialization ---
dbfs_values = []
processed_files = 0
skipped_files = 0
file_details = [] # Optional: Store filename along with dBFS

# --- Input Validation ---
target_dir = Path(wav_dir)
if not target_dir.is_dir():
    print(f"Error: Directory not found: {wav_dir}")
    exit() # Stop execution if directory doesn't exist

# --- Find WAV Files (Case-Insensitive) ---
# Combine results from searching for .wav and .WAV
wav_files = list(target_dir.glob('*.wav')) + list(target_dir.glob('*.WAV'))

if not wav_files:
    print(f"No .wav or .WAV files found in '{wav_dir}'")
    exit()

print(f"Found {len(wav_files)} WAV files. Processing...")

# --- Process Each File ---
for filepath in wav_files:
    try:
        # Read audio data as float64. soundfile typically normalizes to [-1.0, 1.0]
        # Using dtype='float64' ensures high precision for calculations
        data, samplerate = sf.read(filepath, dtype='float64')

        # Handle empty files or potential read issues
        if not isinstance(data, np.ndarray) or data.size == 0:
             print(f"Warning: Skipping '{filepath.name}' - Empty or invalid audio data.")
             skipped_files += 1
             continue

        # Calculate RMS (Root Mean Square) value of the signal
        # np.mean handles multi-channel audio correctly by averaging over all samples
        rms = np.sqrt(np.mean(data**2))

        # Calculate dBFS (decibels relative to full scale)
        # Full scale for float data normalized to [-1.0, 1.0] corresponds to an RMS of 1.0 / sqrt(2) for a sine wave,
        # but for general signals, dBFS is usually 20 * log10(RMS). An RMS of 1.0 for a square wave would be 0 dBFS.
        # We need to handle the case where RMS is 0 (silence) to avoid log10(0) error.
        if rms < 1e-10:  # Use a small threshold to consider it silent
            # dbfs = -np.inf # Mathematically correct, but can be inconvenient
            dbfs = -120.0  # Assign a very low dB value for silence practical plotting/stats
            print(f"Info: '{filepath.name}' is silent or near-silent (RMS < 1e-10). Assigning {dbfs:.1f} dBFS.")
        else:
            dbfs = 20 * np.log10(rms)
            # Clamp dBFS at 0, as RMS shouldn't exceed 1.0 for normalized data
            # (Though floating point inaccuracies might slightly exceed it)
            dbfs = min(dbfs, 0.0)


        dbfs_values.append(dbfs)
        file_details.append({'filename': filepath.name, 'dbfs': dbfs}) # Store details
        processed_files += 1

    except Exception as e:
        print(f"Error processing '{filepath.name}': {e}")
        skipped_files += 1

print(f"\nFinished processing.")
print(f"Successfully processed: {processed_files} files.")
if skipped_files > 0:
    print(f"Skipped due to errors or empty data: {skipped_files} files.")

# --- Calculate Statistics & Plot (only if data exists) ---
if not dbfs_values:
    print("\nNo valid dBFS values were calculated. Cannot generate statistics or plot.")
else:
    dbfs_array = np.array(dbfs_values)

    # Calculate statistics
    min_dbfs = np.min(dbfs_array)
    max_dbfs = np.max(dbfs_array)
    mean_dbfs = np.mean(dbfs_array)

    # --- Output Results ---
    print("\n--- Sound Power Statistics (dBFS) ---")
    print(f"Min dBFS:  {min_dbfs:.2f}")
    print(f"Max dBFS:  {max_dbfs:.2f}")
    print(f"Mean dBFS: {mean_dbfs:.2f}")

    # Optional: Print files with min/max values
    min_file = next((item['filename'] for item in file_details if np.isclose(item['dbfs'], min_dbfs)), "N/A")
    max_file = next((item['filename'] for item in file_details if np.isclose(item['dbfs'], max_dbfs)), "N/A")
    print(f"File with Min dBFS: {min_file}")
    print(f"File with Max dBFS: {max_file}")


    # --- Plot Histogram ---
    plt.figure(figsize=(10, 6))
    # 'auto' automatically determines a reasonable number of bins
    plt.hist(dbfs_array, bins='auto', color='skyblue', edgecolor='black')
    plt.xlabel('Sound Power (dBFS)')
    plt.ylabel('Number of Files')
    plt.grid(axis='y', alpha=0.75)

    # Add vertical lines for mean, min, max
    plt.axvline(mean_dbfs, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {mean_dbfs:.2f}')
    plt.axvline(min_dbfs, color='green', linestyle='dotted', linewidth=1.5, label=f'Min: {min_dbfs:.2f}')
    plt.axvline(max_dbfs, color='purple', linestyle='dotted', linewidth=1.5, label=f'Max: {max_dbfs:.2f}')
    plt.legend()

    plt.tight_layout() # Adjust layout to prevent labels overlapping
    plt.show()