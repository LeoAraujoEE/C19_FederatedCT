import os
import cv2
import numpy as np
import seaborn as sns
import colorcet as cc
import matplotlib.pyplot as plt
sns.set()

from sklearn.metrics import auc
from sklearn.metrics import roc_curve

class CustomPlots:
    def __init__( self, model_path ):

        # Dir where the plots will be stored
        self.plot_dir = os.path.join( os.path.dirname(model_path), "plots" )
        
        # Creates the plot directory if needed
        if not os.path.exists( self.plot_dir ):
            os.makedirs( self.plot_dir )

        return

    def plot_train_results( self, history, dataset_name, figsize = (6, 9) ):

        # Defines the path to the plot directory inside the model's directory
        dst_dir = os.path.join( self.plot_dir, "1.Training Results" )
        
        # Creates the plot directory if needed
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

        for metric in [ "loss", "acc", "f1"]:

            # Defines the plot name
            plot_name = "{}_{}.png".format( metric, dataset_name )
            plot_path = os.path.join( dst_dir, plot_name )

            # Extracts values from history dict
            val_values = history["val_"+metric]
            train_values = history[metric]
            all_values = list(val_values) + list(train_values)

            # Epochs
            epochs = range(1, len(train_values) + 1)

            if metric != "loss":
                leg_loc = "lower right"
                best_x = np.argmax(val_values)
                y_lims = (np.min( [0.7, .95 * np.min( all_values )] ), 1.0)
                
            else:
                leg_loc = "upper right"
                best_x = np.argmin(val_values)
                y_lims = (0, np.min( [1.5, np.max( all_values )] ))

            # Plots the results
            plt.ioff()
            fig = plt.figure( figsize = figsize )
            plt.plot( epochs, train_values, "r", label = "Training" )
            plt.plot([epochs[best_x]], [train_values[best_x]], 
                      color = "r", marker = "o")
            plt.plot( epochs, val_values, "b", label = "Validation" )
            plt.plot([epochs[best_x]], [val_values[best_x]], 
                      color = "b", marker = "o")
            plt.title( metric.title()+" per Epoch", fontsize = 24 )
            plt.xlabel( "Epochs", fontsize = 20 )
            plt.xticks( fontsize = 16 )
            plt.ylabel( metric.title(), fontsize = 20 )
            plt.legend( loc = leg_loc, fontsize = 20 )
            plt.ylim( y_lims )


            # Saves & closes figure
            fig.savefig( plot_path, dpi = 100, bbox_inches = "tight" )
            plt.close( fig )

        return
    
    def plot_fl_global_results( self, history_df, figsize = (16, 12), prefix = "full" ):
        # Converts dataframe to dict
        history = history_df.to_dict("list")

        # Defines the path to the plot directory inside the model's directory
        dst_dir = os.path.join( self.plot_dir, "1.Training Results", "global" )
        
        # Creates the plot directory if needed
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

        for metric in ["loss", "acc", "f1"]:

            # Defines the plot name
            plot_name = f"{prefix}_global_{metric}.png"
            plot_path = os.path.join( dst_dir, plot_name )
            
            # Extracts train values from history dict
            min_train_values = history[f"min_{metric}"]
            avg_train_values = history[f"avg_{metric}"]
            max_train_values = history[f"max_{metric}"]
            
            # Extracts validation values from history dict
            min_val_values = history[f"min_val_{metric}"]
            avg_val_values = history[f"avg_val_{metric}"]
            max_val_values = history[f"max_val_{metric}"]

            # Steps.Epochs
            xticks_labels = [str(i) for i in history["Step.Epoch"]]
            vline_idxs = [i for i, v in enumerate(xticks_labels) if ".0" in v]
            steps_epochs = range(len(xticks_labels))

            # Min/Max values to set plot's ylim
            min_value = np.min(list(min_val_values) + list(min_train_values))
            max_value = np.max(list(max_val_values) + list(max_train_values))

            if metric != "loss":
                # Label for min/max curves and ylims for both plots
                min_label, max_label = "Worst Client", "Best Client"
                y_lims = (np.min( [0.7, .95 * min_value] ), 1.0)
                
                # Coordinates for point of best val performance
                best_x = np.argmax(avg_val_values)
                
            else:
                # Label for min/max curves and ylims for both plots
                min_label, max_label = "Best Client", "Worst Client"
                y_lims = (0, np.min( [1.5, max_value] ))
                
                # Coordinates for point of best val performance
                best_x = np.argmin(avg_val_values)
            
            # Sets the colors for min/max curves
            min_color = "r" if metric != "loss" else "g"
            max_color = "r" if metric == "loss" else "g"

            # Plots the results
            plt.ioff()
            fig, axes = plt.subplots(1, 2, squeeze = False, figsize = figsize)
            
            # Plots min/avg/max curves for training metrics
            plt.sca(axes.flat[0])
            plt.plot(steps_epochs, min_train_values, color = min_color, 
                     linewidth = 1, linestyle = "--", label = min_label)
            plt.plot(steps_epochs, avg_train_values, color = "b", 
                     label = "Average Client")
            plt.plot(steps_epochs, max_train_values, color = max_color, 
                     linewidth = 1, linestyle = "--", label = max_label)
            plt.plot([steps_epochs[best_x]], [avg_train_values[best_x]], 
                      color = "blue", marker = "o")
            plt.title(f"Training {metric.title()} x Step.Epoch", fontsize=24)
            plt.xlabel("Steps.Epochs", fontsize=20 )
            plt.xticks( steps_epochs, xticks_labels, fontsize=16 )
            plt.ylabel( metric.title(), fontsize=20 )
            plt.ylim(y_lims)
            plt.vlines(x=vline_idxs, ymin = y_lims[0], ymax=y_lims[1],
                       color = "k", linestyles="--", linewidth = 1, 
                       label = "Aggregation")
            # plt.legend(loc = leg_loc, fontsize=20)
            
            # Plots min/avg/max curves for training metrics
            plt.sca(axes.flat[1])
            plt.plot(steps_epochs, min_val_values, color = min_color, 
                     linewidth = 1, linestyle = "--", label = min_label)
            plt.plot(steps_epochs, avg_val_values, color = "b", 
                     label = "Average Client")
            plt.plot(steps_epochs, max_val_values, color = max_color, 
                     linewidth = 1, linestyle = "--", label = max_label)
            plt.plot([steps_epochs[best_x]], [avg_val_values[best_x]], 
                     color = "blue", marker = "o")
            plt.title(f"Validation {metric.title()} x Step.Epoch",fontsize=24)
            plt.xlabel("Steps.Epochs", fontsize=20 )
            plt.xticks(steps_epochs, xticks_labels, fontsize=16 )
            plt.ylabel(metric.title(), fontsize=20 )
            plt.ylim(y_lims)
            plt.vlines(x = vline_idxs, ymin = y_lims[0], ymax=y_lims[1],
                       color = "k", linestyles="--", linewidth = 1, 
                       label = "Aggregation")
            # plt.legend(loc = leg_loc, fontsize=20)
            
            handles, labels = axes.flat[1].get_legend_handles_labels()
            fig.legend( handles, labels, ncol = len(labels), 
                        loc = "lower center", fontsize = 18 )

            # Saves & closes figure
            fig.savefig( plot_path, dpi = 100, bbox_inches = "tight" )
            plt.close( fig )

        return
    
    def plot_fl_local_results( self, history_df, client_datasets, figsize = (16, 12), prefix = "full" ):
        # Converts dataframe to dict
        history = history_df.to_dict("list")

        # Defines the path to the plot directory inside the model's directory
        dst_dir = os.path.join( self.plot_dir, "1.Training Results", "local" )
        
        # Creates the plot directory if needed
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
            
        for client_id, dset_name in client_datasets.items():
            client_name = f"client_{client_id}"

            for metric in ["loss", "acc", "f1"]:

                # Defines the plot name
                plot_name = f"{prefix}_{client_name}_local_{metric}.png"
                plot_path = os.path.join( dst_dir, plot_name )
                
                # Extracts train and val values from history dict
                train_values = history[f"{client_name}_{metric}"]
                val_values = history[f"{client_name}_val_{metric}"]

                # Steps.Epochs
                xticks_labels = [str(i) for i in history["Step.Epoch"]]
                vline_idxs = [i for i, v in enumerate(xticks_labels) if ".0" in v]
                steps_epochs = range(len(xticks_labels))

                # Min/Max values to set plot's ylim
                min_value = np.min(list(val_values) + list(train_values))
                max_value = np.max(list(val_values) + list(train_values))

                if metric != "loss":
                    leg_loc = "lower right"
                    best_x = np.argmax(val_values)
                    y_lims = (np.min( [0.7, .95 * min_value] ), 1.0)
                else:
                    leg_loc = "upper right"
                    best_x = np.argmin(val_values)
                    y_lims = (0, np.min( [1.5, max_value] ))

                # Plots the results
                plt.ioff()
                fig = plt.figure( figsize = figsize )
                
                # Plots min/avg/max curves for training metrics
                plt.plot(steps_epochs, train_values, "r", label = "Training")
                plt.plot([steps_epochs[best_x]], [train_values[best_x]], 
                          color = "r", marker = "o")
                plt.plot(steps_epochs, val_values, "b", label = "Validation")
                plt.plot([steps_epochs[best_x]], [val_values[best_x]], 
                          color = "b", marker = "o")
                plt.title(f"{metric.title()} x Step.Epoch - Client_" +
                          f"{client_id} ({dset_name.title()})", fontsize=24)
                plt.xlabel("Steps.Epochs", fontsize=20 )
                plt.xticks( steps_epochs, xticks_labels, fontsize=16 )
                plt.ylabel( metric.title(), fontsize=20 )
                plt.ylim(y_lims)
                plt.vlines(x=vline_idxs, ymin = y_lims[0], ymax=y_lims[1],
                        color = "k", linestyles="--", linewidth = 1, 
                        label = "Aggregation")
                plt.legend(loc = leg_loc, fontsize=20)

                # Saves & closes figure
                fig.savefig( plot_path, dpi = 100, bbox_inches = "tight" )
                plt.close( fig )

        return

    def plot_test_results( self, results, dataset_name, cval_dataset_names = None, figsize = (16, 16) ):

        # Defines the path to the plot directory inside the model's directory
        dst_dir = os.path.join( self.plot_dir, "2.Test Results" )
        
        # Creates the plot directory if needed
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        
        min_value = np.inf

        for metric in [ "acc", "f1", "auc" ]:

            # Defines the plot name
            plot_name = "{}_{}.png".format( metric, dataset_name )
            plot_path = os.path.join( dst_dir, plot_name )

            fig, axes = plt.subplots(2, 1, squeeze=False, figsize=figsize)

            # Plot 1 - Final values for Train, Validation and Test
            labels = [] # Labels for the x-axis in the bar plot
            values = [] # Values for the y-axis in the bar plot
            for partition in ["train", "val", "test"]:
                # Key to result dict
                key = "{}_{}".format(partition, metric)
            
                # Pair of label & value
                labels.append( partition.title() )
                values.append( float(results[key]) )

            # Color palette
            colors = list(sns.color_palette(cc.glasbey, n_colors = len(labels)))

            plt.sca(axes.flat[0])
            bars = plt.barh( labels, values, color = colors, height = 0.2 )
            plt.bar_label(bars)
            plt.title("{} per Partition".format(metric.title()), fontsize = 24)
            plt.yticks(fontsize = 16, rotation = 0)
            axes.flat[0].invert_yaxis()
            plt.xlabel(metric.title(), fontsize = 20)
            min_value = np.min( [min_value, 0.7, .95*np.min( values )] )

            # Plot 2 - Final values for Test & Cross-Val Datasets
            labels = [] # Labels for the x-axis in the bar plot
            values = [] # Values for the y-axis in the bar plot
            dset_names = ["test"] if cval_dataset_names is None else ["test"]+cval_dataset_names
            for name in (dset_names):
                # Key to result dict
                key = "{}_{}".format(name.lower().replace(" ", ""), metric)
                name = name.lower().replace(" ", "") if name != "test" else dataset_name.lower().replace(" ", "")
            
                # Pair of label & value
                labels.append( name.title() )
                values.append( float(results[key]) )

            # Color palette
            colors = list(sns.color_palette(cc.glasbey, n_colors = len(labels)))

            plt.sca(axes.flat[1])
            bars = plt.barh( labels, values, color = colors, height = 0.4 )
            plt.bar_label(bars)
            plt.title("{} per Dataset".format(metric.title()), fontsize = 24)
            plt.yticks(fontsize = 16, rotation = 0)
            axes.flat[1].invert_yaxis()
            plt.xlabel(metric.title(), fontsize = 20)
            min_value = np.min( [min_value, 0.7, .95*np.min( values )] )
            axes.flat[0].set_xlim( (min_value, 1.0) )
            axes.flat[1].set_xlim( (min_value, 1.0) )
            
            fig.savefig( plot_path, bbox_inches = "tight" )
            plt.close(fig)

        return

    def plot_roc_curve( self, true_labels, scores, dataset_name, partition ):
        
        # Defines the path to the plot file inside the model's directory
        dst_dir = os.path.join( self.plot_dir, "3.ROC Curves" )
        roc_fname = "roc_{}_{}.png".format( dataset_name, partition )
        roc_fpath = os.path.join( dst_dir, roc_fname )
        
        # Creates the plot directory if needed
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

        # Compute ROC curve and ROC area for each class
        fpr, tpr, _ = roc_curve(true_labels, scores)
        roc_auc  = auc(fpr, tpr)

        # Plots the confusion matrix as a heatmap
        plt.ioff()
        fig = plt.figure( figsize = (8, 12) )
        ax = plt.gca()

        # Plots ROC Curve for COVID-19
        ax.plot( fpr, tpr, color = "blue", lw = 2, label = "COVID-19 (AUROC = {:.4f})".format(roc_auc))

        # Draws a reference line for a useless classifier in each plot
        ax.plot( [0, 1], [0, 1], "k--", lw = 2 )

        # Sets the limits for each axis
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])

        # Names each axis
        ax.set_ylabel("True Positive Rate")
        ax.set_xlabel("False Positive Rate")
        ax.legend(loc = "lower right")

        # Sets the plot title
        if partition.lower() == "test":
            ax.set_title("ROC Curves: {}".format(dataset_name.title()))
        else:
            ax.set_title("ROC Curves: {} Dataset ({})".format(dataset_name.title(), partition.title()))

        # Saves & closes figure
        fig.savefig( roc_fpath, dpi = 100, bbox_inches = "tight" )
        plt.close( fig )

        return
    
    def plot_confusion_matrix( self, conf_matrix, dataset_name, partition, labels ):
        # Defines the path to the plot file inside the model's directory
        dst_dir = os.path.join( self.plot_dir, "4.Confusion Matrix" )
        cm_fname = "cm_{}_{}.png".format( dataset_name, partition )
        cm_fpath = os.path.join( dst_dir, cm_fname )
        
        # Creates the plot directory if needed
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

        # Normalizes the confusion matriz by dividing each row for its sum, each 
        # element is divided by the total amount of true samples for the true class
        n_rows, n_cols = conf_matrix.shape[:2]
        total_true_counts = np.sum( conf_matrix, axis = 1 ).reshape( n_rows, 1 )
        normalized_cm = conf_matrix / total_true_counts

        # Prepares annotations (counts and percentages) for the plot function
        cm_counts = ["{0:0.0f}".format(c) for c in conf_matrix.flatten()]
        cm_percts = ["{0:.1%}".format(p) for p in normalized_cm.flatten()]
        cm_annots = ["{}\n{}".format(c, p) for c, p in zip(cm_counts, cm_percts)]
        cm_annots = np.asarray(cm_annots).reshape( n_rows, n_cols )

        # Plots the confusion matrix as a heatmap
        plt.ioff()
        plt_h, plt_w = int(2*n_rows), int(2*n_cols)
        fig = plt.figure( figsize = (plt_h, plt_w) )
        blues_cmap = sns.color_palette("Blues", as_cmap=True)
        ax  = sns.heatmap( normalized_cm, annot = cm_annots, fmt="", cmap = blues_cmap, cbar = True, 
                           xticklabels = labels, yticklabels = labels, vmin = 0, vmax = 1)

        # Adds extra information on each label and a title
        ax.set_ylabel("True Labels")
        ax.set_xlabel("Predicted Labels")
        if partition.lower() == "test":
            ax.set_title("{} Dataset".format(dataset_name.title()))
        else:
            ax.set_title("{} Dataset ({})".format(dataset_name.title(), partition.title()))

        # Saves & closes figure
        plt.sca( ax )
        plt.savefig( cm_fpath, dpi = 100, bbox_inches = "tight" )
        plt.close( fig )

        return

    @staticmethod
    def plot_batch( datagen, batch, n_cols, figsize = (16, 9) ):
        assert batch < len(datagen), f"\nInvalid batch num {batch} received. Datagen can only supply {len(datagen)} batches..."
        filenames = datagen.get_fnames()
        csv_classes = datagen.get_labels()

        print(f"\nPlotting batch: {batch}/{len(datagen)}")
        inputs, outputs = datagen.__getitem__(batch)

        batchsize = inputs.shape[0]
        n_rows = np.ceil(batchsize / n_cols).astype(int)
        fig, axs = plt.subplots( nrows = n_rows, ncols = n_cols, figsize = figsize )

        for i in range(batchsize):
            
            # Info from CSV
            filename = filenames[i]
            csv_class = int(csv_classes[i])

            # Input from Datagen
            spec_arr  = inputs[i]
            spec_arr = (spec_arr - np.min(spec_arr)) / np.max(spec_arr - np.min(spec_arr))

            # Output from Datagen
            datagen_class = int(outputs[i] > 0.5)
            datagen_label = datagen.class2label_dict[datagen_class]

            print(f"\tFile: '{filename}', Shape: {spec_arr.shape}, min: {np.min(spec_arr):.4f}, avg: {np.mean(spec_arr):.4f}, max: {np.max(spec_arr):.4f}")

            axs.flat[i].imshow(spec_arr, cmap = "gray", vmin=0, vmax=1)
            axs.flat[i].set_title(f"Label: {datagen_label} - Class: {datagen_class}/{csv_class}", size=11 )
            axs.flat[i].set_xlabel(f"File: '{filename}'\nShape: {spec_arr.shape}", size=11 )
            axs.flat[i].set_xticks([])
            axs.flat[i].set_yticks([])
            axs.flat[i].grid(False)

        fig.tight_layout()
        try:
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
        except:
            pass
        plt.show()
        return fig
