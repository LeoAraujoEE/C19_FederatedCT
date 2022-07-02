import tensorflow as tf
from tensorflow.keras import backend as K

def f1(y_true, y_pred):    
    # source: https://neptune.ai/blog/implementing-the-macro-f1-score-in-keras

    def recall_m(y_true, y_pred):

        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        
        recall = TP / (Positives+K.epsilon())    
        return recall 
    
    
    def precision_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    
        precision = TP / (Pred_Positives+K.epsilon())
        return precision 
    
    recall = recall_m(y_true, y_pred)
    precision = precision_m(y_true, y_pred)
    
    return 2*((precision*recall)/(precision+recall+K.epsilon()))