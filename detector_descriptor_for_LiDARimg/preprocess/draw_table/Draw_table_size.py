# Import the necessary libraries
import os
import pandas as pd

# 512-4096 step:128  original: 2048 we choose to show: 512 640 896 1024 1280 1664 1920 2048  2560 3328 3584 4096 
# 32 - 256 step:32    original: 128  we choose to show:  32 64 128 192 256 


# Initialize an empty DataFrame to store the final means for each size
final_means_df_all = pd.DataFrame()

# Define the directory to search
directory = '/home/jimmy/Downloads/Performance_comparison/preprocess/csv_output'

# Define the sizes you are interested in
sizes = ['512x32','896x128','896x256','1024x64','1024x128','1024x256',
            '1280x64','1280x128','1280x256','1920x128','1920x256','2048x128','2048x256','2560x128','2560x256','4096x128']


# Loop over all sizes
for size in sizes:
    # Initialize an empty DataFrame to store the means for this size
    df_means = pd.DataFrame()

    # Loop over all files in the directory
    for filename in os.listdir(directory):
        # Check if the file is a .csv and contains the current size in its name
        if filename.endswith(".csv") and size in filename :
            print(filename)
            # Read the csv file
            df = pd.read_csv(os.path.join(directory, filename))

            # Calculate the mean of each column and append it to the df_means DataFrame
            df_means = df_means.append(df.mean(), ignore_index=True)

    # Calculate the mean of each column in the df_means DataFrame
    final_means = df_means.mean()
    final_variances = df_means.var()  # Calculate variance
    # Round the means and variances to 3 decimal places
    final_means = final_means.round(3)
    final_variances = final_variances.round(3)

    combined_series = final_means.astype(str) + "|" + final_variances.astype(str)

#---------------------------------------------------------------------------------------------

    # Convert the combined series to a DataFrame and transpose it for better readability
    final_means_var_df = pd.DataFrame(combined_series).transpose()

    # Rename the index
    final_means_var_df = final_means_var_df.rename(index={0: size})

    # Append the final means and variances DataFrame for this size to the overall final means DataFrame
    final_means_df_all = final_means_df_all.append(final_means_var_df)


# Delete some columns
final_means_df_all.drop(['image_number',"evaluation_1",'evaluation_2_1','evaluation_2_2','evaluation_2_3',"evaluation_4",'evaluation_6'], axis=1, inplace=True)

#Define a dictionary with old column names as keys and new column names as values
column_names = { "evaluation_3_1": "Robustness of rotation", "evaluation_3_2": "Robustness of scaling",\
                "evaluation_3_3": "Robustness with noise", "evaluation_5": "Distinctiveness",   "evaluation_7": "Matching Score"}

#Rename the columns
final_means_df_all.rename(columns=column_names, inplace=True)

#使用+来拼接文件路径
file_path = '/home/jimmy/Downloads/Performance_comparison/preprocess/table_about_size.csv'

final_means_df_all.to_csv(file_path)

