import tensorflow as tf
from tensorflow.keras import backend as K

def f1(y_true, y_pred):    
    # based on: https://neptune.ai/blog/implementing-the-macro-f1-score-in-keras

    TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    Positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    
    recall = TP / (Positives+K.epsilon())
    precision = TP / (Pred_Positives+K.epsilon())
    
    return 2*((precision*recall)/(precision+recall+K.epsilon()))