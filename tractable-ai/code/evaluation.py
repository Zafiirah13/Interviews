from sklearn.metrics import matthews_corrcoef, classification_report, confusion_matrix, balanced_accuracy_score
from imblearn.metrics import classification_report_imbalanced, geometric_mean_score
from plot import plot_confusion_matrix
import numpy as np

def evaluation_metrics(y_true, y_pred, classes = ['Repair', 'Replace']):
    '''
    INPUTS:
        y_true (vector): The ground truth label [0, 0 , 1, 1, 0, ...]
        y_pred (vector): The prediction of the classifier [0, 0 , 1, 1, 0, ...]
        
    OUTPUTS:
        metrics (float): accuracy, mcc, precision, recall ....
    '''
    
    balance_accuracy = balanced_accuracy_score(y_true, y_pred)
    conf_mat         = confusion_matrix(y_true, y_pred)
    MCC              = matthews_corrcoef(y_true, y_pred)
    report           = classification_report_imbalanced(y_true, y_pred, target_names = classes)

    plot_confusion_matrix(conf_mat, classes_types=classes, normalize=False) 
    
    print('The balanced-accuracy of the model is {}'.format(balance_accuracy))
    print('The Matthews correlation coefficient of the model is {}\n\n'.format(MCC))
    print(report)
    return balance_accuracy, MCC, report, conf_mat


def to_labels(probs, threshold):
    '''
    Function to convert probabilities to labels based on specific threshold
    INPUTS:
        probs (arr)      : An array of probability varying from {0,1}
        threshold (float): The threshold chosen (0.5, 0.6, ..etc)
        
    OUTPUT:
        threshold (arr): An array with labels [0, 1, 0, 1, 1, ...]
    '''
    return (probs >= threshold).astype('int')

def optimal_threshold_tuning(y_prob, y_true, thresholds):
    '''
    Funtion to find the optimal threshold based on two metrics: Balanced accuracy and Geometric mean scores
    
    INPUTS:
        y_prod (arr)    : An array of probabilities varying from {0,1}
        y_true (arr)    : An array of labels [0, 1, 0, 1, ...]
        thresholds (arr): An array of thresholds [0.01, 0.2, 0.3, 0.4, ...]
        
   OUTPUTS:     
        accuracy (arr): An array of balanced accuracy calculated based on different threshold
        gmeans   (arr): An array of geometric mean score calculated based on different threshold
    '''
    accuracy = [balanced_accuracy_score(y_true, to_labels(y_prob, t)) for t in thresholds]
    gmeans_score = [geometric_mean_score(y_true, to_labels(y_prob, t)) for t in thresholds]
    ix = np.argmax(accuracy); iy = np.argmax(gmeans_score)
    print('Optimal Threshold = %.3f based on Balanced Accuracy Score = %.3f' % (thresholds[ix], accuracy[ix]))
    print('Optimal Threshold = %.3f based on GMeans-Score = %.3f' % (thresholds[iy], gmeans_score[iy]))
    return accuracy, gmeans_score
