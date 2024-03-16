import os, re, shutil
import pandas as pd
# Requires pandas, openpyxl

# Generates one csv file for each animal subset (Fish, Bird...)
# Splits the training and validation set as such so we can train individually

# Split either train, test, or val (mode).
def split_csv(csv_path, xlsx_path, mode):
    # Load CSV and XLSX files into pandas
    csv_data = pd.read_csv(csv_path, delimiter=';')
    xlsx_data = pd.read_excel(xlsx_path)

    # Iterate over each row of the input XLSX (AR_metadata)
    for index, row in xlsx_data.iterrows():
        video_id = row['video_id']

        # Format string entry of animal classes into a list
        animal_parent_classes = re.findall(r"'(.*?)'", row['list_animal_parent_class'])
        
        # Iterate over each animal class that appears in the video
        for animal_parent_class in animal_parent_classes:
            # Construct output file path
            output_file_path = f"{mode}_splits/{animal_parent_class.replace(' ', '-')}.csv"
            
            # Check if the output file already exists, if not create it
            if not os.path.exists(output_file_path):
                with open(output_file_path, 'w'):
                    pass  # Create an empty file
                    
            # Filter train_light.csv data based on video_id and write to the output file
            filtered_data = csv_data[csv_data['video_id'] == video_id]
            with open(output_file_path, 'a') as f:
                filtered_data.to_csv(f, index=False, header=f.tell()==0, sep=';')  # Append to the file if it already exists

                
# Delete existing output
if os.path.isdir('val_splits'):
    shutil.rmtree('val_splits')
    
if os.path.isdir('train_splits'):
    shutil.rmtree('train_splits')

os.mkdir('val_splits')
os.mkdir('train_splits')

# Paths to input files
train_csv_path = 'train_light.csv'
val_csv_path = 'val_light.csv'
xlsx_path = '../AR_metadata.xlsx'

# Split the training data
split_csv(train_csv_path, xlsx_path, 'train')
# Split the val data (testing?)
split_csv(val_csv_path, xlsx_path, 'val')
