import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
sns.set()

from IPython.display import display_html
from itertools import chain,cycle

def get_color_dict():
    # Loads two palettes from seaborn
    hls_palette    = sns.color_palette("hls", 8)
    paired_palette = sns.color_palette("Paired", 12)
    set_2_palette  = sns.color_palette(  "Set2", 12)
    
    # Creates a dictionary that associates colors for each possible value inside the plots
    color_dict = { "M":           paired_palette[0], "F":                     paired_palette[4], "Axial":           paired_palette[0],
                   "Yes":         paired_palette[2], "No":                    paired_palette[4], "CT":              paired_palette[0],
                   "train":       paired_palette[0], "val":                   paired_palette[2], "test":            paired_palette[4],
                   "Expert":      paired_palette[0], "Automatic":             paired_palette[2], "Non-expert":      paired_palette[4],
                   "Normal":      paired_palette[0], "COVID-19":              paired_palette[4], "Pneumonia":       paired_palette[2],
                   "China":       paired_palette[5], "Iran":                  paired_palette[3], "Australia":       paired_palette[1],    
                   "Italy":       paired_palette[2], "Algeria":                set_2_palette[1], "Belgium":         paired_palette[10],  
                   "England":     paired_palette[9], "Scotland":              paired_palette[0], "Turkey":          paired_palette[4], 
                   "Azerbaijan":  set_2_palette[2],  "Lebanon":                set_2_palette[3], "Ukraine":          set_2_palette[5],  
                   "Afghanistan": paired_palette[7], "Peru":                  paired_palette[6], "France":           hls_palette[5],
                   "USA":            hls_palette[4], "N/A":                    set_2_palette[7],
                   "CNCB":        paired_palette[0], "iCTCF":                 paired_palette[2], "COVID-CTset":     paired_palette[4],
                   "TCIA":        paired_palette[5], "COVID-19-20 Challenge": paired_palette[6], "radiopaedia.org": paired_palette[7],
                   "LIDC-IDRI":   paired_palette[8], "coronacases.org":       paired_palette[10],"COVID-CT-MD":      set_2_palette[2], 
                   "STOIC":       paired_palette[1], "Stony Brook":           paired_palette[3], "COVID-19-CT-Seg":  set_2_palette[5], 
                   "Known Ages":  paired_palette[0],
                   "Others":      paired_palette[11], "All": paired_palette[8]
                 }
    
    return color_dict

def get_unq_ages(df):
    
    # Computes the number of known ages and the number of unknown ages
    n_unknown_ages = len(df[df["age"] == "N/A"])
    n_known_ages   = len(df[df["age"] != "N/A"])
    
    if (n_known_ages == 0) and (n_unknown_ages > 0):
        unique_entries = ["N/A"]
        entry_counts   = [n_unknown_ages]
        
        return unique_entries, entry_counts
    
    unique_entries = ["N/A", "Known Ages"]
    entry_counts   = [n_unknown_ages, n_known_ages]
        
    return unique_entries, entry_counts

def turn_small_slices_to_others( uniques, counts ):
    # Iterates through unique labels and checks which ones correspond to less of 0.5% of rows
    others   = [ u for u, c in zip(uniques, counts) if c / np.sum(counts) < 0.005 ]
    
    # If there arent less than 2, returns the original lists
    if len(others) < 2:
        return uniques, counts
    
    # Otherwise, removes all labels from others and adds a new label "Others"
    new_unqs = [ u for u in uniques if not u in others ] + ["Others"]
    new_cts  = [ c for u, c in zip(uniques, counts) if not u in others]
    new_cts  = new_cts + [np.sum(counts) - np.sum(new_cts)]
    
    # Then, returns the new lists
    return new_unqs, new_cts
    

def column_as_pie( df, column, partition = None, ax = None ):
    
    if len(df) == 0:
        return
    
    if column.lower() != "age":
        # Gets the unique elements from the specified column and their respective counts
        unq_entries, entrie_cts = np.unique( df[column].to_list(), return_counts = True )
        
        # If there are labels whose relative frequency is lower than 0.5%, they are turned into a new label "Others" for the plot
        unq_entries, entrie_cts = turn_small_slices_to_others( unq_entries, entrie_cts )
        
    else:
        unq_entries, entrie_cts = get_unq_ages(df)    
    
    # Generates a dictionary to define colors used for each unique label
    color_dict = get_color_dict()
    
    # Orders the labels accordingly to the order they appear in color_dict
    label_order = [entry for entry in color_dict.keys() if entry in unq_entries]
    unq_entries, entrie_cts = zip(*sorted(zip(unq_entries, entrie_cts), key=lambda x: label_order.index(x[0])))
    
    # Lists the selected colors for the labels in the current plot
    sel_colors = [ color_dict[entry] for entry in unq_entries ]
    
    # If an axis was passed
#     if not ax is None:
        # Sets it as the current axis
#         plt.sca( ax )
    
    # Plots theextracted data as a pie chart
    wedges, labels, autopct = ax.pie( entrie_cts, labels = unq_entries, startangle = 0, colors = sel_colors, 
                                      autopct = "%1.1f%%", pctdistance = 0.8, textprops={'fontsize': 16}, 
                                      labeldistance = 1.15, rotatelabels = False, counterclock = False )
        
    plot_title = "Complete Dataset" if partition is None else "{} Partition".format( partition.title() if partition.lower() != "val" else "Validation" )
    ax.set_title( plot_title, fontsize = 20)
    
    return

def column_per_partition_as_pie( df, column, dataset, figure, title_complement = ""):
    
    if dataset != "COVIDx CT-2A":
        df = df[df["source"] == dataset]

    axes = figure.subplots( 1, 4 )

    # Plots the distribution for the whole dataset to the first axis
    column_as_pie( df, column, ax = axes[0] )
    
    # Plots the distribution for different partitions in the other 3 axis
    for idx, part in enumerate( ["train", "val", "test"] ):
        column_as_pie( df[df["partition"] == part], column, partition = part, ax = axes[idx+1] )
    
    fig_suptitle = title_complement + " Distribution per {}".format(column.title().replace("_"," "))
    figure.suptitle( fig_suptitle, fontsize = 24, y = 1.005 )
    
    return

def column_histogram( df, column, partition = None, ax = None ):
    
    # If an axis was passed
    if not ax is None:
        # Sets it as the current axis
        plt.sca( ax )
    
    if column.lower() == "sample_count":
        bins = np.arange(0, 850, 25)
        data = df[column].to_list()
        
    elif column.lower() == "age":
        bins = np.arange(0, 101, 5)
        data = df[ df["age"] != "N/A" ]["age"].to_list()
        
    plt.yscale("log")  
    plt.ylabel("Counts (Log Scale)")
        
    labels = ["[{}-{})".format(bins[i], bins[i+1]) for i in range(len(bins)-1)]
    
    plt.hist(data, bins, label=labels, rwidth=0.9)
    plt.xticks(bins[:-1] + 2.5, labels)
    
    plot_title = "Complete Dataset" if partition is None else "{} Partition".format( partition.title() if partition.lower() != "val" else "Validation" )
    plt.title( plot_title, fontsize = 20)
    
    return

def column_per_partition_as_histogram( df, column, dataset, figure, title_complement = ""):
    
    # Generates a dictionary to define colors used for each unique label
    color_dict = get_color_dict()
    
    count_df = column_bins_per_partition_as_df( df, column, dataset )
    count_df.drop(index=("SUM"), inplace = True)

    axes = figure.subplots( 4, 1 )
    
    count_df.plot.bar(y = "Dataset", rot = 0, ax = axes[0], color = color_dict["All"])
    max_count = count_df["Dataset"].max()
    max_count = max_count if max_count > 1 else 10
    
    axes[0].set_ylim([1, 10**np.ceil(np.log10(max_count))])
    axes[0].set_yscale("log")  
    axes[0].set_ylabel("Counts (Log Scale)")
    for p in axes[0].patches:
        axes[0].annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
    
    # Plots the distribution for different partitions in the other 3 axis
    for idx, part in enumerate( ["train", "val", "test"] ):
        count_df.plot.bar(y = part.title(), rot = 0, ax = axes[1+idx], color = color_dict[part])
        axes[1+idx].set_ylim([1, 10**np.ceil(np.log10(max_count))])
        axes[1+idx].set_yscale("log")  
        axes[1+idx].set_ylabel("Counts (Log Scale)")
        for p in axes[1+idx].patches:
            axes[1+idx].annotate(str(p.get_height()), (p.get_x() * 1.000, p.get_height() * 1.005))
            
    fig_suptitle = "{} Distribution ".format(column.title()) + title_complement
    fig_suptitle = title_complement + " Distribution per {}".format(column.title().replace("_"," "))
    
    sup_title_y = 0.9 if column.lower() == "sample_count" else 1.003
    figure.suptitle( fig_suptitle, fontsize=24, y = sup_title_y )
    
    return

def column_per_partition_as_df( df, column, dataset ):
    
    if len(df) == 0:
        return
    
    if column.lower() != "age":
        # Gets the unique elements from the specified column and their respective counts
        unq_entries, entrie_cts = np.unique( df[column].to_list(), return_counts = True )
    
        # Sorts those elements based on their counts
        unq_entries, entrie_cts = zip(*sorted(zip(unq_entries, entrie_cts), key = lambda x: x[1], reverse = True))
        
    else:
        unq_entries, entrie_cts = get_unq_ages(df)    
    
    if dataset != "COVIDx CT-2A":
        df = df[df["source"] == dataset]
    
    count_dict = { "Values": list(unq_entries), "Dataset": [], "Train": [], "Val": [], "Test": [] }
    
    # 
    for idx, key in enumerate( ["Dataset", "Train", "Val", "Test"] ):
        #
        sub_df = df if key.lower() == "dataset" else df[df["partition"] == key.lower()]
        
        for unq in unq_entries:
            #
            if unq != "Known Ages":
                count = len(sub_df[ sub_df[column] == unq ])
                
            else:
                count = len(sub_df[ sub_df[column] != "N/A" ])
                
            count_dict[key].append(count)
    
    count_dict["Values"].append( "SUM" )
    count_dict["Dataset"].append( np.sum(count_dict["Dataset"]) )
    count_dict["Train"].append( np.sum(count_dict["Train"]) )
    count_dict["Val"].append( np.sum(count_dict["Val"]) )
    count_dict["Test"].append( np.sum(count_dict["Test"]) )
    
    dst_df = pd.DataFrame.from_dict( count_dict )
    dst_df.set_index("Values", drop=True, inplace=True)
    return dst_df

def column_bins_per_partition_as_df( df, column, dataset ):
    
    if dataset != "COVIDx CT-2A":
        df = df[df["source"] == dataset]
    
    if column.lower() == "sample_count":
        bins = np.arange(0, 850, 25)
        
    elif column.lower() == "age":
        bins = np.arange(0, 101, 5)
    
    # Removes N/A entries and casts any string values to numeric values (i.e. '25.0' -> 25)
    tmp_df = df[ df[column] != "N/A" ].copy(deep = True)
    tmp_df[column] = pd.to_numeric(tmp_df[column], downcast="integer")
    
    bins_as_tuples = [ (bins[i], bins[i+1]) for i in range(len(bins) - 1) ]
    bins_as_string = [ "[{}, {})".format(bin[0], bin[1]) for bin in bins_as_tuples ]
    count_dict = { "Values": bins_as_string, "Dataset": [], "Train": [], "Val": [], "Test": [] }
    
    # 
    for idx, key in enumerate( ["Dataset", "Train", "Val", "Test"] ):
        #
        sub_df = tmp_df if key.lower() == "dataset" else tmp_df[tmp_df["partition"] == key.lower()]
        
        for unq in bins_as_tuples:
            #
            start_val, end_val = unq
            count = len(sub_df[ (sub_df[column] >= start_val) & (sub_df[column] < end_val)])
            count_dict[key].append(count)
    
    count_dict["Values"].append( "SUM" )
    count_dict["Dataset"].append( np.sum(count_dict["Dataset"]) )
    count_dict["Train"].append( np.sum(count_dict["Train"]) )
    count_dict["Val"].append( np.sum(count_dict["Val"]) )
    count_dict["Test"].append( np.sum(count_dict["Test"]) )
    
    dst_df = pd.DataFrame.from_dict( count_dict )
    dst_df.set_index("Values", drop=True, inplace=True)
    return dst_df

def convert_df_sample2patient( df ):
    # Gets the unique patient_ids and their respective sample count
    unique_patients, sample_count = np.unique( df["patient_id"].to_list(), return_counts = True )
    
    # Creates a dictionary to access sample counts by the patient_id
    patient2sample_dict = { u: c for u, c in zip(unique_patients, sample_count) }
    
    # Copies sample dataframe to build patient dataframe
    dst_df = df.copy(deep = True)
    
    # Drops columns realated to specific samples
    dst_df.drop( ["filename", "x_min", "y_min", "x_max", "y_max"], inplace=True, axis=1 )
    
    # Drops duplicated rows to keep a single row per patient_id
    dst_df.drop_duplicates(subset=None, keep="first", inplace=True)
    
    # Resets Index
    dst_df = dst_df.reset_index(drop=True)
    
    # Creates a new column with the amount of samples available per patient
    dst_df["sample_count"] = dst_df.apply(lambda x: patient2sample_dict[x["patient_id"]], axis=1)
    
    return dst_df

# TODO: Comentar isso aqui
def column_as_pies( s_df, p_df, column, dataset, figsize = (48, 24), resplit = False ):
    
    fig = plt.figure(constrained_layout = True, figsize = figsize )
#     fig.suptitle('Figure title', fontsize=24)

    subfigs = fig.subfigures(nrows=2, ncols=1)
    
    complement = "Sample" if not resplit else "Old Sample"
    column_per_partition_as_pie( s_df, column, dataset, subfigs[0], title_complement = complement)
    
    complement = "Patient" if not resplit else "New Sample"
    column_per_partition_as_pie( p_df, column, dataset, subfigs[1], title_complement = complement)
    
    return

def column_as_histograms( s_df, p_df, column, dataset, figsize = (48, 24), resplit = False ):
    
    fig = plt.figure(constrained_layout = True, figsize = figsize )
#     fig.suptitle('Figure title', fontsize=24)

    subfigs = fig.subfigures(nrows=1, ncols=2)
    
    complement = "Sample" if not resplit else "Old Sample"
    column_per_partition_as_histogram( s_df, column, dataset, subfigs[0], title_complement = complement)
    
    complement = "Patient" if not resplit else "New Sample"
    column_per_partition_as_histogram( p_df, column, dataset, subfigs[1], title_complement = complement)
    
    return

def display_side_by_side(*args,titles=cycle([''])):
    # source: https://stackoverflow.com/questions/38783027/jupyter-notebook-display-two-pandas-tables-side-by-side
    html_str=''
    for df,title in zip(args, chain(titles,cycle(['</br>'])) ):
        html_str+='<th style="text-align:center"><td style="vertical-align:top">'
        html_str+=f'<h2>{title}</h2>'
        html_str+=df.to_html().replace('table','table style="display:inline"')
        html_str+='</td></th>'
    display_html(html_str,raw=True)

def column_as_table( s_df, p_df, column, dataset, bin_flag = False, resplit = False ):
    
    if not ( (bin_flag) and (column.lower() == "age") ):
        s_counts_df = column_per_partition_as_df( s_df, column, dataset )
        p_counts_df = column_per_partition_as_df( p_df, column, dataset )
    
    else:
        s_counts_df = column_bins_per_partition_as_df( s_df, column, dataset )
        p_counts_df = column_bins_per_partition_as_df( p_df, column, dataset )

    complement = "Sample" if not resplit else "Old Sample"
    s_table_title = "{} Distribution\n by {}".format(complement, column.title().replace("_"," "))
    
    complement = "Patient" if not resplit else "New Sample"
    p_table_title = "{} Distribution\n by {}".format(complement, column.title().replace("_"," "))
    display_side_by_side(s_counts_df, p_counts_df, titles=[s_table_title, p_table_title])
    return
