import os
import itertools
import numpy as np
import pandas as pd

def gen_partition_groups( age_bin_len = None ):
    
    # Age bins
    if age_bin_len is None:
        age_bins = list(np.arange(0, 101, 5))
    else:
        age_bins = list(np.arange(0, 101, age_bin_len))
        
    
    # Possible values for all varying metadata
    possible_values_dict = { "age":             [ "N/A" ] + [ (age_bins[i], age_bins[i+1]) for i in range(len(age_bins)-1) ],
                             "sex":             [ "N/A", "M", "F" ],
                             "class":           [ "Normal", "Pneumonia", "COVID-19" ],
                             "slice_selection": [ "N/A", "Expert", "Automatic", "Non-expert" ],
                             "country":         [ "N/A", "China", "Iran", "Australia", "Italy", 
                                                  "Algeria", "Belgium", "England", "Scotland", 
                                                  "Turkey", "Azerbaijan", "Lebanon", "Ukraine", 
                                                  "Afghanistan", "Peru" ]
                           }
    
    # Makes a list with all lists of possible values to compute all possible combinations
    lists_of_possible_values = [ possible_values_dict[key] for key in possible_values_dict.keys() ]
    
    # Uses itertools to compute all possible value combinations and stores it to a list of tuples
    group_list = list( itertools.product( *lists_of_possible_values ) )
    
    # Generates a dictionary from the list of tuples in order to build a dataframe
    group_dict = { key: [g[idx] for g in group_list] for idx, key in enumerate(possible_values_dict.keys()) }
    
    # Splits (start_age, final_age) tuples to make two different columns
    group_dict["start_age"] = [g[0][0] if isinstance(g[0], tuple) else "N/A" for g in group_list ]
    group_dict["final_age"] = [g[0][1] if isinstance(g[0], tuple) else "N/A" for g in group_list ]
    
    # Creates the dataframe
    group_df = pd.DataFrame.from_dict(group_dict)
    
    return group_df

def assign_group( row, g_df, col_list = None ):
    
    # Creates a copy of the group dataframe, which has all possible combination for all columns considered in the data split
    sub_g_df = g_df.copy(deep = True)
    
    if col_list is None:
        # Lists of columns to use while filtering sub_g_df
        col_list = [ "sex", "class", "slice_selection", "country", "age" ]
    
    row_values = []
    # If the patient's age is known
    if row["age"] != "N/A":
        # Converts it from str to int
        age = int(float(row["age"]))
        # Changes col_list to replace "age" by "start_age" as "final_age" since patient's age is binned
        col_list = [ i for i in col_list if i != "age" ]
        col_list = col_list + [ "start_age", "final_age" ]
        # Removes all rows where "start_age" or "final_age" is unknown since the patient's age is known
        sub_g_df = sub_g_df[ (sub_g_df["start_age"] != "N/A") & (sub_g_df["final_age"] != "N/A") ]
        row_values = [age]
    
    # Gradually filters sub_g_df based on columns from col_list to find the correct group for the current patient
    for col in col_list:
        # Filters all rows where the current column's value doesnt match the patient's value 
        # unless the current column is "start_age" or "final_age"
        if not col in [ "start_age", "final_age" ]:
            sub_g_df = sub_g_df[ sub_g_df[col] == row[col] ]
            row_values.append(row[col])
        
        elif col == "start_age":
            # Filters all rows where the starting age for the bin is higher than the patient's age 
            sub_g_df = sub_g_df[ sub_g_df[col] <= age ]
        
        elif col == "final_age":
            # Filters all rows where the final age for the bin is lower than the patient's age 
            sub_g_df = sub_g_df[ sub_g_df[col] > age ]
        
        else:
            # This line should never be executed
            print( "Unknown column '{}'...".format(col) )
    
    # It's expected a single row will only match a single group
    indexes = sub_g_df.index.to_list()
    assert len(indexes) == 1, "Found {} groups -> {}\n{}\n{}\n{}".format(len(indexes), row_values, col_list, row, sub_g_df.head())
    
    # This group's index is then returned
    return indexes[0]
    
def select_patients( df, n_samples ):
    
    # Removes all patientes whose sample_count exceeds the desired number of samples
    g_p_df = df[df["sample_count"] <= n_samples].copy(deep = True)
    
    if len(g_p_df) == 0:
        return [], 0
    
    # Creates a new column with the cumulative sum of the sample_counts for each patient
    g_p_df["sample_count_cumsum"] = g_p_df["sample_count"].cumsum()
    
    # Filters g_p_df to select as many patients as possible so that 
    # the total number of samples is less than or equal to n_samples
    filtered_p_df = g_p_df[ g_p_df["sample_count_cumsum"] <= n_samples ]
    
    # Gets the number of selected samples and the list of selected patients
    n_selected_samples    = filtered_p_df["sample_count_cumsum"].max()
    selected_patient_list = filtered_p_df["patient_id"].to_list()
    
    # Computes the number of missing samples and filters g_p_df to remove selected patients
    n_missing_samples = (n_samples - n_selected_samples)
    filtered_p_df = g_p_df[ ~(g_p_df["sample_count_cumsum"] <= n_samples) ]
    
    # Recursively selects more patients until either find n_samples or there are no more rows to select
    xtra_sel_patients, n_xtra_sel_samples = select_patients( filtered_p_df, n_missing_samples )
    
    # Combines the selected patient_ids and the number of selected samples
    n_selected_samples    = n_selected_samples + n_xtra_sel_samples
    selected_patient_list = selected_patient_list + xtra_sel_patients
    
    # Returns the selected patient_list and the total of selected samples
    return selected_patient_list, n_selected_samples
    

def dataset_by_samples( s_df, p_df, dataset, age_bin_len = None, sample_frac = 0.2, seed = 25 ):
    """ Remakes the split of the datasets to create a validation and a test set.
    """
    
    # Generates a dataframe with all possible value combinations for all varying metadata and associates a number to each combination
    group_df = gen_partition_groups( age_bin_len = age_bin_len )
    
    # Copies s_df and p_df while filtering their rows to keep only rows regarding the current dataset
    sub_sample_df  = s_df[ s_df["source"] == dataset ].copy( deep = True )
    sub_patient_df = p_df[ p_df["source"] == dataset ].copy( deep = True )
    
    # Resets the index for both dataframes
    sub_sample_df.reset_index(drop = True, inplace = True)
    sub_patient_df.reset_index(drop = True, inplace = True)
    
    # Moves all samples to the train partition
    sub_sample_df["partition"] = "train"
    sub_patient_df["partition"] = "train"
    
    # Creates a new column to assign a group to each patient in the dataset
    # This group refers to the specific combination of values for the metadata considered by group_df
    sub_patient_df["group"] = sub_patient_df.apply( lambda x: assign_group( x, group_df ), axis = 1 )
    
    # Replicates the group column to the sample dataframe
    id2group_dict = { row["patient_id"]: row["group"] for idx, row in sub_patient_df.iterrows() }
    sub_sample_df["group"] = sub_sample_df.apply( lambda x: id2group_dict[x["patient_id"]], axis = 1 )
    
    # Shuffles sub_patient_df according to the provided seed
    sub_patient_df = sub_patient_df.sample(frac=1, random_state = seed).reset_index(drop=True)
    
    # Computes the unique groups for this dataset and their respective number of patients
    unq_groups = np.unique( sub_patient_df["group"].to_list() )
    
    selected_val_patient_ids = []
    selected_test_patient_ids = []
    for g_idx, group in enumerate(unq_groups):
        group_as_tuple = tuple([ group_df.iloc[group, :][col] for col in [ "sex", "class", "slice_selection", "country", "age" ] ])
        print("{}/{}: Group {}".format( str(g_idx+1).rjust(5), len(unq_groups), group_as_tuple ))
    
        # Copies sub_sample_df and sub_patient_df while filtering their rows to keep only rows regarding the current group
        group_s_df  = sub_sample_df[ sub_sample_df["group"] == group ].copy( deep = True )
        group_p_df  = sub_patient_df[ sub_patient_df["group"] == group ].copy( deep = True )

        # Resets the index for both dataframes
        group_s_df.reset_index(drop = True, inplace = True)
        group_p_df.reset_index(drop = True, inplace = True)
        
        # Computes the number of samples from the current group to move to test/val partition
        # Only n_test_samples is computed as idealy n_val_samples should be equal to n_test_samples
        n_test_samples = int(sample_frac * len(group_s_df))
        
        selected_patients, n_selected_samples = select_patients( group_p_df, n_test_samples )
        print("\t[Test] Moved {} samples ({} patients), expected {} samples, had {} samples ({} patients)...".format( n_selected_samples,
                                                                                                              len(selected_patients),
                                                                                                              n_test_samples,
                                                                                                              len(group_s_df),
                                                                                                              len(group_p_df) ))
        print("\t\tList of sample counts:", group_p_df["sample_count"].to_list() )
        
        # Adds selected patients to the selected val list
        selected_test_patient_ids = selected_test_patient_ids + selected_patients
    
        # Copies sub_sample_df and sub_patient_df while filtering their rows to keep only rows regarding the current group
        group_s_df  = group_s_df[ ~(group_s_df["patient_id"].isin(selected_patients)) ]
        group_p_df  = group_p_df[ ~(group_p_df["patient_id"].isin(selected_patients)) ]

        # Resets the index for both dataframes
        group_s_df.reset_index(drop = True, inplace = True)
        group_p_df.reset_index(drop = True, inplace = True)
        
        selected_patients, n_selected_samples = select_patients( group_p_df, n_test_samples )
        print("\t[Val] Moved {} samples ({} patients), expected {} samples, had {} samples ({} patients)...".format( n_selected_samples,
                                                                                                             len(selected_patients),
                                                                                                             n_test_samples,
                                                                                                             len(group_s_df),
                                                                                                             len(group_p_df) ))
        print("\t\tList of sample counts:", group_p_df["sample_count"].to_list() )
        
        # Adds selected patients to the selected val list
        selected_val_patient_ids = selected_val_patient_ids + selected_patients
        print("\n\n")
    
    # Changes the partition of the rows whose patient_id is in either selected_test_patient_ids or selected_val_patient_ids
    sub_sample_df["partition"]  = sub_sample_df.apply(lambda x: "test" if x["patient_id"] in selected_test_patient_ids else ( "val" if x["patient_id"] in selected_val_patient_ids else "train"), axis = 1 )
    sub_patient_df["partition"] = sub_patient_df.apply(lambda x: "test" if x["patient_id"] in selected_test_patient_ids else ( "val" if x["patient_id"] in selected_val_patient_ids else "train"), axis = 1 )
    
    
    sub_sample_df.drop(["group"], inplace = True, axis=1)
    sub_patient_df.drop(["group"], inplace = True, axis=1)
    
    return sub_sample_df, sub_patient_df

# ------------------------------------------------------------------------------------------------------------------------------------------

def radiopaedia_dataset_by_samples( s_df, p_df, dataset, sample_frac = 0.2, seed = 20 ):
    """ Remakes the split of the datasets to create a validation and a test set.
    """
    
    # Generates a dataframe with all possible value combinations for all varying metadata and associates a number to each combination
    group_df = gen_partition_groups( age_bin_len = 20 )
    
    # Copies s_df and p_df while filtering their rows to keep only rows regarding the current dataset
    sub_sample_df  = s_df[ s_df["source"] == dataset ].copy( deep = True )
    sub_patient_df = p_df[ p_df["source"] == dataset ].copy( deep = True )
    
    # Resets the index for both dataframes
    sub_sample_df.reset_index(drop = True, inplace = True)
    sub_patient_df.reset_index(drop = True, inplace = True)
    
    # Moves all samples to the train partition
    sub_sample_df["partition"] = "train"
    sub_patient_df["partition"] = "train"
    
    # Copies sub_sample_df and sub_patient_df before combining some of the groups to facilitate data split
    preserved_sample_df = sub_sample_df.copy( deep = True )
    preserved_patient_df = sub_patient_df.copy( deep = True )
    
    # Sets all age, sex and country values to "N/A"
    # Since this dataset has a huge variety of metadata, but very few patients for each combination
    # it was needed to focus on the most important feature to balance (class), so the others were disregarded
    # However, a decent split was still produced by experimenting with different seed values (best found was seed = 20)
    sub_patient_df["age"] = "N/A"
    sub_patient_df["sex"] = "N/A"
    sub_patient_df["country"] = "N/A"
    
    # Combines Non-Expert samples with N/A to facilitate data split
    
    # Creates a new column to assign a group to each patient in the dataset
    # This group refers to the specific combination of values for the metadata considered by group_df
    sub_patient_df["group"] = sub_patient_df.apply( lambda x: assign_group( x, group_df ), axis = 1 )
    
    # Replicates the group column to the sample dataframe
    id2group_dict = { row["patient_id"]: row["group"] for idx, row in sub_patient_df.iterrows() }
    sub_sample_df["group"] = sub_sample_df.apply( lambda x: id2group_dict[x["patient_id"]], axis = 1 )
    
    # Shuffles sub_patient_df according to the provided seed
    sub_patient_df = sub_patient_df.sample(frac=1, random_state = seed).reset_index(drop=True)
    
    # Computes the unique groups for this dataset and their respective number of patients
    unq_groups = np.unique( sub_patient_df["group"].to_list() )
    
    selected_val_patient_ids = []
    selected_test_patient_ids = []
    for g_idx, group in enumerate(unq_groups):
        group_as_tuple = tuple([ group_df.iloc[group, :][col] for col in [ "sex", "class", "slice_selection", "country", "age" ] ])
        print("{}/{}: Group {}".format( str(g_idx+1).rjust(5), len(unq_groups), group_as_tuple ))
    
        # Copies sub_sample_df and sub_patient_df while filtering their rows to keep only rows regarding the current group
        group_s_df  = sub_sample_df[ sub_sample_df["group"] == group ].copy( deep = True )
        group_p_df  = sub_patient_df[ sub_patient_df["group"] == group ].copy( deep = True )

        # Resets the index for both dataframes
        group_s_df.reset_index(drop = True, inplace = True)
        group_p_df.reset_index(drop = True, inplace = True)
        
        # Computes the number of samples from the current group to move to test/val partition
        # Only n_test_samples is computed as idealy n_val_samples should be equal to n_test_samples
        n_test_samples = int(sample_frac * len(group_s_df))
        
        selected_patients, n_selected_samples = select_patients( group_p_df, n_test_samples )
        print("\t[Test] Moved {} samples ({} patients), expected {} samples, had {} samples ({} patients)...".format( n_selected_samples,
                                                                                                              len(selected_patients),
                                                                                                              n_test_samples,
                                                                                                              len(group_s_df),
                                                                                                              len(group_p_df) ))
        print("\t\tList of sample counts:", group_p_df["sample_count"].to_list() )
        
        # Adds selected patients to the selected val list
        selected_test_patient_ids = selected_test_patient_ids + selected_patients
    
        # Copies sub_sample_df and sub_patient_df while filtering their rows to keep only rows regarding the current group
        group_s_df  = group_s_df[ ~(group_s_df["patient_id"].isin(selected_patients)) ]
        group_p_df  = group_p_df[ ~(group_p_df["patient_id"].isin(selected_patients)) ]

        # Resets the index for both dataframes
        group_s_df.reset_index(drop = True, inplace = True)
        group_p_df.reset_index(drop = True, inplace = True)
              
        selected_patients, n_selected_samples = select_patients( group_p_df, n_test_samples )
        print("\t[Val] Moved {} samples ({} patients), expected {} samples, had {} samples ({} patients)...".format( n_selected_samples,
                                                                                                             len(selected_patients),
                                                                                                             n_test_samples,
                                                                                                             len(group_s_df),
                                                                                                             len(group_p_df) ))
        print("\t\tList of sample counts:", group_p_df["sample_count"].to_list() )
        
        # Adds selected patients to the selected val list
        selected_val_patient_ids = selected_val_patient_ids + selected_patients
        print("\n\n")
    
    # Changes the partition of the rows whose patient_id is in either selected_test_patient_ids or selected_val_patient_ids
    preserved_sample_df["partition"]  = preserved_sample_df.apply(lambda x: "test" if x["patient_id"] in selected_test_patient_ids else ( "val" if x["patient_id"] in selected_val_patient_ids else "train"), axis = 1 )
    
    preserved_patient_df["partition"] = preserved_patient_df.apply(lambda x: "test" if x["patient_id"] in selected_test_patient_ids else ( "val" if x["patient_id"] in selected_val_patient_ids else "train"), axis = 1 )
    
    return preserved_sample_df, preserved_patient_df

# ------------------------------------------------------------------------------------------------------------------------------------------

def update_metadata_csv( csv_path, s_df, dataset, save_bool = False ):
    s_df = s_df[ s_df["source"] == dataset].reset_index(drop = True)
    
    # If the csv already exists
    if os.path.exists( csv_path ):

        # Reads the csv file
        df = pd.read_csv(csv_path, sep = ";", na_filter = False, dtype={"age": str})

        # Removes any entries from this dataset to avoid duplicates
        df = df[ ~(df["source"] == dataset) ]

        # Concatenates the new split data and new_split_df
        new_split_df = pd.concat([ df, s_df], ignore_index = True )

    else:
        new_split_df = s_df
    
    if save_bool:
        # Saves the dataframe
        new_split_df.to_csv( csv_path, index = False, sep = ";" )
    
    return new_split_df
