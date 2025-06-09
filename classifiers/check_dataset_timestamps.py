import os
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np  # Useful for calculating the mean timestamp

# --- Configuration ---
root_dir = '/Volumes/Rectangle/little_owl/data-backup/Little Owl AudioMoth'
# Define the expected timestamp format in the filename
# Example: 20241210_213000.WAV -> YYYYMMDD_HHMMSS
timestamp_format = "%Y%m%d_%H%M%S"
folders_to_exclude = ["AudioMoth test"]  # Add folders to exclude from analysis
# --- End Configuration ---

file_data=[]
skipped_files = 0

print(f"Scanning directory: {root_dir}")

# Walk through the directory tree
for subdir, dirs, files in os.walk(root_dir):
    if os.path.basename(os.path.dirname(subdir)) in folders_to_exclude:
        continue
    print(f" Processing folder: {os.path.relpath(subdir, root_dir)}...") # Show relative path
    for filename in files:
        # Check if the file has a . wav extension (case-insensitive)
        if filename.lower().endswith('.wav'):
            # Extract the base name without extension
            base_name = os.path.splitext(filename)[0]
            full_filepath = os.path.join(subdir, filename) # Get the full path
            try:
                # Attempt to parse the filename as a datetime object
                dt_object = datetime.datetime.strptime(base_name, timestamp_format)
                file_data.append((dt_object, full_filepath)) # Store both
            except ValueError:
                # Handle filenames that don't match the expected format
                # print(f"  - Warning: Could not parse timestamp from filename: {filename}")
                skipped_files += 1

print(f"\nScan complete. Found {len(file_data)} valid timestamped WAV files.")
if skipped_files > 0:
    print(f"Skipped {skipped_files} files due to non-matching name format.")

# Proceed only if file_data were found
if not file_data:
    print("\nNo valid timestamped WAV files found. Cannot generate statistics or plot.")
else:
    # --- Find Min/Max and Calculate Statistics ---

    # Sort the list based on the datetime object (the first element of the tuple)
    file_data.sort(key=lambda item: item[0])

    earliest_time, earliest_filepath = file_data[0]     # First element after sorting
    latest_time, latest_filepath = file_data[-1]    # Last element after sorting

    # Extract just the timestamps for mean calculation and histogram
    timestamps = [item[0] for item in file_data]

    # Calculate mean time
    numeric_timestamps = [dt.timestamp() for dt in timestamps]
    mean_numeric_timestamp = np.mean(numeric_timestamps)
    mean_time = datetime.datetime.fromtimestamp(mean_numeric_timestamp)

    # --- Print Statistics ---
    print("\n--- Time Range Statistics ---")
    print(f"Earliest file timestamp: {earliest_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Located at: {earliest_filepath}") # Print location
    print(f"Latest file timestamp:   {latest_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Located at: {latest_filepath}")   # Print location
    print(f"Mean file timestamp:     {mean_time.strftime('%Y-%m-%d %H:%M:%S')}")
    time_span = latest_time - earliest_time
    print(f"Total time span covered: {time_span}")


    # --- Create Histogram ---
    print("\nGenerating histogram...")

    # Determine number of bins - adjust as needed for better visualization
    num_bins = min(50, len(timestamps) // 10) if len(timestamps) > 20 else 10 # Heuristic
    num_bins = max(5, num_bins) # Ensure at least 5 bins

    plt.figure(figsize=(12, 6))

    plt.hist(timestamps, bins=num_bins, edgecolor='black')

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=10))

    plt.gcf().autofmt_xdate()

    plt.xlabel("Timestamp")
    plt.ylabel("Number of Files")
    plt.title(f"Distribution of AudioMoth Recording Timestamps\n(Directory: {os.path.basename(root_dir)})")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    print("Histogram displayed.")
