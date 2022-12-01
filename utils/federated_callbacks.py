import os
import glob
import json
import time
import shutil
import random
import logging
import warnings
import subprocess
import numpy as np
import pandas as pd

class FederatedCallback():
    def __init__(self, monitor_var = None, min_delta = 0):
        # Selected monitored variable
        self.monitor_var = monitor_var
        self.min_delta = min_delta
        
        if isinstance(self.monitor_var, str):
            self.mode = "max" if not "loss" in self.monitor_var else "min"
            self.best_val = -np.inf if self.mode == "max" else np.inf
            self.prev_best = -np.inf if self.mode == "max" else np.inf
            
        return
    
    def has_improved(self, val):
        assert (not self.monitor_var is None)
        
        # Otherwise, creates checkpoint if monitor_best has improved
        if self.mode == "max":
            improved_bool = (val > self.best_val)
        else:
            improved_bool = (val < self.best_val)
            
        # Updates best val in case of improvement
        improvement = 0
        if improved_bool:
            improvement = np.abs(val - self.best_val)
            self.prev_best = self.best_val
            self.best_val = val
            
        # Only returns True if the improvement is larger than min_delta
        return (improvement > np.abs(self.min_delta))
 
class FederatedModelCheckpoint(FederatedCallback):
    def __init__(self, ckpt_weights_path, monitor_var = None):
        super().__init__(monitor_var, min_delta = 0)
        
        # Path to checkpoint weights file
        self.ckpt_weights_path = ckpt_weights_path
            
        return
    
    def create_checkpoint(self, val, src_weights_path):
        if (self.monitor_var is None) or (self.has_improved(val)):
            print(f"\nGlobal model's '{self.monitor_var}' improved from",
                  f"{self.prev_best:.4f} to {self.best_val:.4f}. Saving",
                  f"model to '{self.ckpt_weights_path}'...")
            self.copy_weights(src_weights_path, self.ckpt_weights_path)
            return
        
        print(f"\nGlobal model's '{self.monitor_var}' did not improve from",
                f"{self.prev_best:.4f}...")
        return
    
    @staticmethod
    def copy_weights(src_weights_path, dst_weights_path):
        assert os.path.exists(src_weights_path)
        
        # Gets the path for the model's configs file
        src_configs_path = src_weights_path.replace(".h5", ".json")
        dst_configs_path = dst_weights_path.replace(".h5", ".json")
        
        # Copies the selected model's weights and configs
        dst_path_list = [dst_weights_path, dst_configs_path]
        src_path_list = [src_weights_path, src_configs_path]
        for src_path, dst_path in zip(src_path_list, dst_path_list):
            if os.path.exists(dst_path):
                os.remove(dst_path)
            shutil.copy2(src_path, dst_path)
        return
    

class FederatedEarlyStopping(FederatedCallback):
    def __init__(self, monitor_var = None, min_delta = 0, 
                 patience_steps = None, patience_epochs = None, 
                 epochs_per_step = None):
        super().__init__(monitor_var, min_delta)
        
        #
        self.best_step = 0
        
        if not patience_steps is None:
            self.patience_steps = patience_steps
            return
        
        assert_str = "".join(["Variables 'patience_epochs' and",
                             "'epochs_per_step' must be provided if",
                             "'patience_steps' is None..."])
        assert not patience_epochs is None, assert_str
        assert not epochs_per_step is None, assert_str
        
        self.patience_steps = np.ceil(patience_epochs / epochs_per_step)
        
        return
    
    def is_triggered(self, val, step_idx):
        # Ignores EarlyStopping if monitored_var is None 
        # or if its patience is set to 0
        if (self.monitor_var is None) or (self.patience_steps == 0):
            return False
        
        # Updates best_step index in case of improvement
        if self.has_improved(val):
            self.best_step = step_idx
            return False

        if (step_idx - self.best_step) >= self.patience_steps:
            print(f"\nGlobal model's '{self.monitor_var}' hasn't improved",
                  f"for {self.patience_steps} steps. Forcing Early Stop...")
            return True
        return False