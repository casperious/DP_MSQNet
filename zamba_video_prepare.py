import pandas as pd
import shutil
import os

def crete_video_folder(csv_path,old_directory, new_directory):

    df = pd.read_csv(csv_path)

    


    if not os.path.exists(new_directory):
        os.makedirs(new_directory)

    # Move the files
    for filepath in df['filepath']:
        old_file_path = os.path.join(old_directory, filepath)
        new_file_path = os.path.join(new_directory, filepath)

        
        if os.path.isfile(old_file_path):
            shutil.copy(old_file_path, new_file_path)

        else:
            print(f'File not found: {filepath}')

if __name__ == '__main__':
    #File path and directory path need to change
    old_directory = '../AnimalKingdom/action_recognition/dataset/video'
    train_directory = '../AnimalKingdom/action_recognition/dataset/zamba_train_video'
    val_directory = '../AnimalKingdom/action_recognition/dataset/zamba_pred_video'
    train_csv = '../zamba_trainV.csv'
    val_csv = '../zamba_valV.csv'
    crete_video_folder(train_csv,old_directory, train_directory)
    crete_video_folder(val_csv,old_directory, train_directory) 