import os
from re import I
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

    def plot_train_results( self, history, dataset_name, fine_tune = False, figsize = (6, 9) ):

        # Defines the path to the plot directory inside the model's directory
        dst_dir = os.path.join( self.plot_dir, "1.Training Results" )
        
        # Creates the plot directory if needed
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

        for metric in [ "loss", "acc", "f1", "auc" ]:

            # Defines the plot name
            plot_name = "{}_{}.png".format( metric, dataset_name )
            plot_path = os.path.join( dst_dir, plot_name )

            # Extracts values from history dict
            val_values = history["val_"+metric]
            train_values = history[metric]
            all_values = list(val_values) + list(train_values)

            # Epochs
            epochs = range(1, len(train_values) + 1)

            # Plots the results
            plt.ioff()
            fig = plt.figure( figsize = figsize )
            plt.plot( epochs, train_values, "r", label = "Training" )
            plt.plot( epochs, val_values, "b", label = "Validation" )
            plt.title( metric.title()+" per Epoch", fontsize = 24 )
            plt.xlabel( "Epochs", fontsize = 20 )
            plt.xticks( fontsize = 16 )
            plt.ylabel( metric.title(), fontsize = 20 )

            if metric != "loss":
                plt.legend( loc = "lower right", fontsize = 20 )
                plt.ylim( (np.min( [0.7, .95*np.min( all_values )] ), 1.0) )
            else:
                plt.legend( loc = "upper right", fontsize = 20 )
                plt.ylim( (0, np.min( [1.5, np.max( all_values )] )) )


            # Saves & closes figure
            fig.savefig( plot_path, dpi = 100, bbox_inches = "tight" )
            plt.close( fig )

        return

    def plot_test_results( self, results, dataset_name, cval_dataset_names = None, figsize = (16, 12) ):

        # Defines the path to the plot directory inside the model's directory
        dst_dir = os.path.join( self.plot_dir, "2.Test Results" )
        
        # Creates the plot directory if needed
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

        for metric in [ "acc", "f1", "auc" ]:

            # Defines the plot name
            plot_name = "{}_{}.png".format( metric, dataset_name )
            plot_path = os.path.join( dst_dir, plot_name )

            fig, axes = plt.subplots(1, 2, squeeze=False, figsize=figsize)

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
            bars = plt.bar( labels, values, color = colors, width = 0.4 )
            plt.bar_label(bars)
            plt.title("{} per Partition".format(metric.title()), fontsize = 24)
            plt.xlabel("Partition", fontsize = 20)
            plt.xticks(fontsize = 16, rotation = 45)
            plt.ylabel(metric.title(), fontsize = 20)
            plt.ylim( (np.min( [0.7, .95*np.min( values )] ), 1.0) )

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
            bars = plt.bar( labels, values, color = colors, width = 0.4 )
            plt.bar_label(bars)
            plt.title("{} per Dataset".format(metric.title()), fontsize = 24)
            plt.xlabel("Dataset", fontsize = 20)
            plt.xticks(fontsize = 16, rotation = 45)
            plt.ylabel(metric.title(), fontsize = 20)
            plt.ylim( (np.min( [0.7, .95*np.min( values )] ), 1.0) )
            fig.savefig( plot_path, bbox_inches = "tight" )
            plt.close(fig)

        return

    def plot_roc_curve( self, true_labels, scores, dataset_name, partition, labels ):
        
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
        ax.set_ylabel("True")
        ax.set_xlabel("Predicted")
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

        print("\nPlotting batch:")
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

            axs.flat[i].imshow(spec_arr, cmap = "magma", vmin=0, vmax=1)
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
