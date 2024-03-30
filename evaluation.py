import torch
from torch import tensor
import pandas as pd
from torchmetrics.classification import MultilabelAveragePrecision
import numpy as np
import main_eval
import subprocess
import re
from ast import literal_eval
import pickle
import warnings
warnings.filterwarnings("ignore")

# Ensure tensors are moved to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def filter_matrix_values_and_indexes(A, threshold):
    # Moving A to the device (GPU if available)
    A = A.to(device)
    filtered_values = []
    indexes = []

    for i in range(A.shape[0]):
        row = A[i]
        above_threshold = row >= threshold

        if above_threshold.any():
            #print("Yes, elements above threshold in row", i)
            filtered_row_values = row[above_threshold].tolist()
            filtered_row_indexes = torch.nonzero(above_threshold, as_tuple=False).squeeze().tolist()
        else:
            #print("No elements above threshold in row", i)
            max_val_index = torch.argmax(row)  # Get the index of the max value in the row
            filtered_row_values = [row[max_val_index].item()]  # Get the max value itself
            filtered_row_indexes = [max_val_index.item()]  # Convert index to item (int)

        filtered_values.append(filtered_row_values)
        indexes.append(filtered_row_indexes)

    return filtered_values, indexes

def eval(pred, target):
    metric = MultilabelAveragePrecision(num_labels=140, average="micro", thresholds=None)
    return metric(pred, target)

def get_weighted_prediction(filtered_prob, msq_pred):
    # Convert filtered_prob and msq_pred to PyTorch tensors, ensure they are of type float32, and move them to the device
    filtered_prob = torch.tensor(filtered_prob, dtype=torch.float32, device=device)
    msq_pred = torch.tensor(msq_pred, dtype=torch.float32, device=device)
    
    result = torch.matmul(filtered_prob, msq_pred)
    # Convert result back to CPU for any further non-GPU operations, if necessary
    return result.cpu().numpy()


def prepare_label():
    df_original = pd.read_csv('.../AnimalKingdom/action_recognition/annotation/val_light.csv',delimiter=';')
    # Expanding the 'labels' column into a list of integers
    df_original['labels'] = df_original['labels'].apply(lambda x: [int(i) for i in x.split(',')])

    # Creating a new DataFrame with video_id and 140 additional columns
    video_ids = df_original['video_id']
    columns = [str(i) for i in range(140)]
    target = pd.DataFrame(0, index=video_ids, columns=columns)
    # Updating the new DataFrame based on the labels
    for index, row in df_original.iterrows():
        for label in row['labels']:
            if label < 140:  # Making sure the label is within the 0-139 range
                target.at[row['video_id'], str(label)] = 1

    target.reset_index(inplace=True)
    target.rename(columns={'index': 'video_id'}, inplace=True)
    return target

def random_sample_zamba(file_path):
    
    df = pd.read_csv(file_path)

    # Randomly sample rows from the DataFrame. Adjust the number of rows as needed.
    sampled_df = df.sample(frac=0.02)  # Here, sampling changed from 0.1

    # Reindex the sampled DataFrame
    sampled_df.reset_index(drop=True, inplace=True)

    return sampled_df


if __name__ == '__main__':

    ori_zamba_file = '.../zamba_predictions.csv'
    

    label_categories = pd.read_csv('.../AnimalKingdom/action_recognition/annotation/df_action.csv')
    head = label_categories.loc[label_categories['segment'] == 'head']
    head = head['index']
    head = head.to_numpy()
    tail = label_categories.loc[label_categories['segment']=='tail']
    tail = tail['index']
    tail = tail.to_numpy()
    middle = label_categories.loc[label_categories['segment']=='middle']
    middle = middle['index']
    middle=  middle.to_numpy()
    print(head,tail,middle)
    species = pd.read_csv(ori_zamba_file)
    #species = random_sample_zamba(ori_zamba_file)
    species_exclude_path = species.iloc[:, 1:]
    species_tensor = torch.tensor(species_exclude_path.to_numpy())

    threshold = 0.9   #changed from 0.1 checking

    filtered_values, indexes = filter_matrix_values_and_indexes(species_tensor, threshold)
    
    dic_species = {0:'Amphibian', 1:'Bird', 2:'Fish', 3:'Insect', 4:'Mammal', 5:'Reptile', 6:'Sea-animal'}
    video_list = species.iloc[:, 0].tolist()
 
    filepath = '.../AnimalKingdom/action_recognition/dataset/image/' 
    command = "python3 main_eval.py --dataset animalkingdom --model timesformerclipinitvideoguide --gpu 0 "
    #After filtering, index will point to which MSQNet shoule be run. for example first video results in indexes: [0, 3],
    # we will need to run amphibian and insect MSQNet.
    #Based on the indexes, we get the prediction result for each video, 
    #Get the final predictions for all 6096 videos
    labels = prepare_label()
    final_predictions = []
    final_heads = []
    final_tails = []
    final_middles = []
    true_labels=[]
    true_heads = []
    true_tails = []
    true_middles = []
    for index in range(len(video_list)):
        #get prediction for a specific MSQNet and concatenate them
        #MSQ_pred is m' by k
        video_name_split = video_list[index].split('/')
        video_name = video_name_split[-1]
        name_split = video_name.split('.')
        vid_name = name_split[0]
        print(vid_name)
        vid_labels = labels.loc[labels['video_id']==vid_name]
        #print(vid_labels,type(vid_labels))
        labs = vid_labels.drop(columns=["video_id"])
        #print(labs,type(labs))
        extracted_labels = []
        for itr in range(140):
            if(labs.iloc[0,itr] == 1):
                extracted_labels.append(itr)
                print(itr)
        print(extracted_labels)
        if extracted_labels[0] in head:
            true_heads.append(vid_labels)
        elif extracted_labels[0] in tail:
            true_tails.append(vid_labels)
        else:
            true_middles.append(vid_labels)
        true_labels.append(vid_labels)
        fp = filepath+vid_name+'/'
        print(true_heads,true_tails,true_middles)
        if(isinstance(indexes[index],list)):
            final_output=[]
            for i in range(len(indexes[index])):
                #print("In multi class")
                val = indexes[index]
                checkpoint = dic_species[val[i]]
#                 print(checkpoint)
                run_command = command + "--weights checkpoint_"+checkpoint +".pth --filepath "+fp
                process = subprocess.check_output(run_command, shell=True, text=True)
                #print(process)
                output = process
                output = output.replace('\n','')
                output = output.replace(' ','')
                #print("step1: ", output)
                output = output[8:-16]
                
                output_arr = np.fromstring(output[1:-1],dtype=np.float64,sep=',')
                #print(output_arr)
                output_arr = output_arr.tolist()
                #print(output_arr)
                final_output.append(output_arr)
            final_arr = np.array(final_output)
         
            res = get_weighted_prediction(filtered_values[index],final_arr)
            if extracted_labels[0] in head:
                final_heads.append(res)
            elif extracted_labels[0] in tail:
                final_tails.append(res)
            else:
                final_middles.append(res)
            final_predictions.append(res)
        else:
            #print("In single class")
            checkpoint = dic_species[indexes[index]]
#             print(checkpoint)
            run_command = command + "--weights checkpoint_"+checkpoint +".pth --filepath "+fp
            process = subprocess.check_output(run_command, shell=True, text=True)
            output = process
            #print(output)
            output = output.replace('\n','')
           # print(output)
            output = output.replace(' ','')
            #print(output)
            output = output[8:-16]
            #print(output)
            output_arr = np.fromstring(output[1:-1],dtype=np.float64,sep=',')
            #print(output_arr)
            output_arr = output_arr.tolist()
            #output_arr = output_arr.reshape((1,140))
            #res = get_weighted_prediction(filtered_values[index],output_arr)
            print(output_arr)
            if extracted_labels[0] in head:
                final_heads.append(output_arr)
            elif extracted_labels[0] in tail:
                final_tails.append(output_arr)
            else:
                final_middles.append(output_arr)
            final_predictions.append(output_arr)


    #print(final_predictions)
    final_predictions_tensor = torch.tensor(final_predictions)
    
    final_heads_tensor = torch.tensor(final_heads)
    final_tails_tensor = torch.tensor(final_tails)
    final_middles_tensor = torch.tensor(final_middles)

    true_labels_processed = [df.drop(columns=["video_id"]).to_numpy() for df in true_labels]
    print(true_labels_processed)
    true_labels_tensor = torch.tensor(true_labels_processed)
    true_labels_tensor = true_labels_tensor.squeeze() 

    true_heads_processed = [df.drop(columns=["video_id"]).to_numpy() for df in true_heads]
    print(true_heads_processed)
    true_heads_tensor = torch.tensor(true_heads_processed)
    true_heads_tensor = true_heads_tensor.squeeze() 
    true_heads_tensor = true_heads_tensor.long()
    print(true_heads_tensor)
    
    true_tails_processed = [df.drop(columns=["video_id"]).to_numpy() for df in true_tails]
    true_tails_tensor = torch.tensor(true_tails_processed)
    true_tails_tensor = true_tails_tensor.squeeze() 
    true_tails_tensor = true_tails_tensor.long()

    
    true_middles_processed = [df.drop(columns=["video_id"]).to_numpy() for df in true_middles]
    true_middles_tensor = torch.tensor(true_middles_processed)
    true_middles_tensor = true_middles_tensor.squeeze()
    true_middles_tensor = true_middles_tensor.long()

    
    metric = eval(final_predictions_tensor,true_labels_tensor) * 100
    metric_heads = eval(final_heads_tensor,true_heads_tensor) * 100
    metric_tails = eval(final_tails_tensor,true_tails_tensor)*100
    metric_middles = eval(final_middles_tensor,true_middles_tensor)*100
    with open('.../full_metric_results.txt', 'w') as f:
        f.write(f'Overall MAP result: {metric}\n Head MAP result: {metric_heads}\n Tail MAP result: {metric_tails}\n Middle MAP result: {metric_middles}')
