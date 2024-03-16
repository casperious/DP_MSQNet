import os, re, shutil
import pandas as pd

def zamba_train(csv_path, xlsx_path):
    # Load CSV and XLSX files into pandas
    csv_data = pd.read_csv(csv_path, delimiter=' ')
    xlsx_data = pd.read_excel(xlsx_path)
 
   

    # Merge 
    merged_df = pd.merge(csv_data, xlsx_data[['video_id', 'list_animal_parent_class']], left_on='original_vido_id', right_on='video_id', how='left')
#     print(merged_df.columns)
    final_df = merged_df[['original_vido_id', 'list_animal_parent_class']]
    final_df.columns = ['filepath', 'label']
    cleaned_df = final_df.drop_duplicates()
    cleaned_df = cleaned_df.reset_index(drop=True)
    cleaned_df['filepath'] = cleaned_df['filepath'] + '.mp4'



    cleaned_df.to_csv('zamba_train.csv', index=False)
    

    

if __name__ == '__main__':
    # Need to change xlsx_path, train_csv_path, and val_csv_path
    xlsx_path = '../AR_metadata.xlsx'
    train_csv_path = '../train.csv'
    val_csv_data = pd.read_csv('../val.csv',delimiter=' ')


    zamba_train(train_csv_path,xlsx_path)

    csv_data = pd.read_csv('zamba_train.csv')

    csv_data['label'] = csv_data['label'].str.strip("[]").str.replace("'", "")

#     print(csv_data.head())
    csv_data['label'] = csv_data['label'].str.split(', ')
    exploded_df = csv_data.explode('label')

#     print(exploded_df.head())
    exploded_df.to_csv('zamba_trainV.csv', index=False)

    
    df_filepath = val_csv_data[['original_vido_id']]
    df_filepath = val_csv_data[['original_vido_id']].drop_duplicates()

    
    df_filepath.rename(columns={'original_vido_id': 'filepath'}, inplace=True)
    df_filepath.reset_index(drop=True, inplace=True)
    df_filepath['filepath'] = df_filepath['filepath'] + '.mp4'
   
    df_filepath.to_csv('zamba_valV.csv', index=False)
