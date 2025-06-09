# this file is for condensing different csvs for paper presentation.

import csv
import argparse

def condense_csvs(csv1_path, csv2_path, output_path="condensed_results.csv"):
    """
    Condense two CSV files into one, extracting specific metrics.
    
    Args:
        csv1_path (str): Path to the first CSV file
        csv2_path (str): Path to the second CSV file
        output_path (str): Path for the output condensed CSV
    """
    # Read the first CSV file
    with open(csv1_path, 'r') as file1:
        csv1_data = list(csv.reader(file1))
        header1 = csv1_data[0]
        rows1 = csv1_data[1:]
    
    # Read the second CSV file
    with open(csv2_path, 'r') as file2:
        csv2_data = list(csv.reader(file2))
        header2 = csv2_data[0]
        rows2 = csv2_data[1:]
    
    # Create output file
    with open(output_path, 'w', newline='') as out_file:
        csv_writer = csv.writer(out_file)
        
        # Write header
        csv_writer.writerow([
            "n_val_1", "mean_rl_auc_1", "std_dev_rl_auc_1", "mean_rl_f1_1", "std_dev_rl_f1_1",
            "n_val_2", "mean_rl_auc_2", "std_dev_rl_auc_2", "mean_rl_f1_2", "std_dev_rl_f1_2"
        ])
        
        # Determine max length to iterate through
        max_rows = max(len(rows1), len(rows2))
        
        # Write data rows
        for i in range(max_rows):
            row = []
            
            # Add data from CSV1 if available
            if i < len(rows1):
                row.extend([
                    rows1[i][0],  # n_val
                    rows1[i][1],  # mean_rl-auc
                    rows1[i][2],  # std_dev_rl-auc
                    rows1[i][3],  # mean_rl-f1
                    rows1[i][4],  # std_dev_rl-f1
                ])
            else:
                row.extend(["", "", "", "", ""])
                
            # Add data from CSV2 if available
            if i < len(rows2):
                row.extend([
                    rows2[i][0],  # n_val
                    rows2[i][1],  # mean_rl-auc
                    rows2[i][2],  # std_dev_rl-auc
                    rows2[i][3],  # mean_rl-f1
                    rows2[i][4],  # std_dev_rl-f1
                ])
            else:
                row.extend(["", "", "", "", ""])
                
            csv_writer.writerow(row)
    
    print(f"Condensed CSV created at {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Condense two CSV files into one for paper presentation.')
    parser.add_argument('csv1_path', help='Path to the first CSV file')
    parser.add_argument('csv2_path', help='Path to the second CSV file')
    parser.add_argument('--output', '-o', default='condensed_results.csv', 
                        help='Output path for the condensed CSV (default: condensed_results.csv)')
    
    args = parser.parse_args()
    
    condense_csvs(args.csv1_path, args.csv2_path, args.output)