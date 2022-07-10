import os
import pandas as pd
from utils.custom_model_trainer import ModelManager

train_dataset = "radiopaedia.org"
reference_dataset = "Comp_CNCB_iCTCF_a_dropped"
reference_metrics = ["test_f1"]

trainManager = ModelManager( train_dataset, keep_pneumonia = True )
trainManager.doJsonSearch( reference_dataset, reference_metrics )
