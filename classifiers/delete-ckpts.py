# CAREFUL! This script deletes files and folders.
# It is recommended to run it with caution and ensure you have backups if necessary.
# This script is designed to delete all .CKPT files and 'deploy' folders (and empty 'Temp' folders) from all experiment subdirectories

import os
import shutil  # For deleting folders

def delete_ckpts_and_deploy(root_directory):
    """
    Deletes .CKPT files from 'results' subdirectories and 'deploy' folders
    within a given root directory.  Prints the names of files/folders that
    would be deleted instead of actually deleting them.

    Args:
        root_directory (str): The path to the root directory containing the
                              subdirectories with 'results' and 'deploy' folders.
    """

    for subdir_name in os.listdir(root_directory):
        subdir_path = os.path.join(root_directory, subdir_name)

        # Check if it's a directory
        if os.path.isdir(subdir_path):
            results_dir_path = os.path.join(subdir_path, 'Results')
            # find deploy folder
            deploy_folder_path = None
            for a_path in os.listdir(subdir_path):
                if a_path.endswith('_Deploy'):
                    deploy_folder_path = os.path.join(subdir_path, a_path)
                    break

            # Delete 'deploy' folder
            if deploy_folder_path:
                if os.path.isdir(deploy_folder_path):
                    print(f"Would delete deploy folder: {deploy_folder_path}")
                    shutil.rmtree(deploy_folder_path)  # Uncomment to actually delete

            # Delete 'Temp' folder
            temp_folder_path = os.path.join(subdir_path, 'Temp')
            if os.path.isdir(temp_folder_path):
                print(f"Would delete Temp folder: {temp_folder_path}")
                shutil.rmtree(temp_folder_path)  # Uncomment to actually delete

            # Delete .CKPT files from 'results' directory
            if os.path.isdir(results_dir_path):
                for filename in os.listdir(results_dir_path):
                    if filename.endswith('.ckpt'):
                        ckpt_file_path = os.path.join(results_dir_path, filename)
                        print(f"Would delete CKPT file: {ckpt_file_path}")
                        os.remove(ckpt_file_path)

# Example Usage:  Replace with your actual root directory
root_dir = "/Users/kaspar/Documents/bioacoustic-data-augmentation/classifiers/evaluation_results"
delete_ckpts_and_deploy(root_dir)