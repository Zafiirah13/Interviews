import matplotlib.pylab as plt
import numpy as np
import itertools

plt.rc('text', usetex=True)
plt.rc('font',**{'family':'DejaVu Sans','serif':['Palatino']})
figSize  = (12, 8)
fontSize = 20

def plot_confusion_matrix(cm, classes_types,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.RdPu):#  plt.cm.Reds):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """


    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    else:
        cm = cm.astype('int')
    plt.figure(figsize=(9,8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    cb=plt.colorbar(fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=16)
    tick_marks = np.arange(len(classes_types))
    plt.xticks(tick_marks, classes_types, rotation=45)
    plt.yticks(tick_marks, classes_types)
    plt.tick_params(axis='x', labelsize=16)
    plt.tick_params(axis='y', labelsize=16)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if (cm[i, j] < 0.01) or (cm[i,j] >= 0.75)  else "black",fontsize=18)
        else:
            plt.text(j, i,"{:0}".format(cm[i, j]), horizontalalignment="center",
                    color="white" if (cm[i, j] < 1000) or (cm[i,j] >= 3000)  else "black",fontsize=18)

    
    plt.ylabel('True label',fontsize = 16)
    plt.xlabel('Predicted label', fontsize = 16)
    plt.tight_layout()


def plot_roc_with_thres(fpr, tpr, auc, thresholds):
    '''
    Function to plot the ROC curves (Receiver Characteristic Curves)
    INPUTS:
        fpr: False Positive rate
        tpr: True Positive Rate
        auc: Area under curve
        ofname: The directory to save the roc curve
    '''
    # Compute gmeans
    gmeans = np.sqrt(tpr * (1-fpr))
    
    # locate the index of the largest g-mean
    ix = np.argmax(gmeans)
    
    print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))
    
    plt.figure(figsize=figSize)
    plt.scatter(fpr[ix], tpr[ix], s=120, marker='o', color='black', label='Best Thresh = {0:0.3f}'.format(thresholds[ix]))
    plt.plot(fpr, tpr, marker='.', label='AUC = {0:0.2f}'.format(auc))
    
    plt.xlabel("False Positive Rate",fontsize=fontSize)
    plt.ylabel("True Positive Rate",fontsize=fontSize)
    plt.ylim([0.0,1.01])
    plt.tick_params(axis='both', labelsize=fontSize)
    plt.legend(loc="best",prop={'size':14},bbox_to_anchor=(1,0.15))
    plt.tight_layout()
    plt.show()

def plot_roc(fpr, tpr, auc):
    '''
    Function to plot the ROC curves (Receiver Characteristic Curves)
    INPUTS:
        fpr: False Positive rate
        tpr: True Positive Rate
        auc: Area under curve
        ofname: The directory to save the roc curve
    '''

    plt.figure(figsize=figSize)
    plt.plot(fpr, tpr, marker='.', label='ROC curve (area = {0:0.2f})'.format(auc))
    plt.xlabel("False Positive Rate",fontsize=fontSize)
    plt.ylabel("True Positive Rate",fontsize=fontSize)
    plt.ylim([0.0,1.01])
    plt.tick_params(axis='both', labelsize=fontSize)
    plt.legend(loc="best",prop={'size':14},bbox_to_anchor=(1,0.15))
    plt.tight_layout()
    plt.show()


def plot_train_val_test_dis(repair_train_distn,replace_train_distn, 
                            repair_val_distn,replace_val_distn,
                            repair_test_distn, replace_test_distn):
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14,4))
    ax1.hist(repair_train_distn, density=True, alpha=0.2, label = ' Train Repair')
    ax1.hist(replace_train_distn, density=True, alpha=0.2, label = ' Train Replace')
    ax1.set_xlabel('URR Classifier Output',fontsize=fontSize)
    ax1.tick_params(axis='both', labelsize=fontSize)
    ax1.set_title('Train Set', fontsize=fontSize)

    ax2.hist(repair_val_distn, density=True, alpha=0.2, label = 'Val Repair')
    ax2.hist(replace_val_distn, density=True, alpha=0.2, label = 'Val Replace')
    ax2.set_xlabel('URR Classifier Output',fontsize=fontSize)
    ax2.tick_params(axis='both', labelsize=fontSize)
    ax2.set_title('Validation Set', fontsize=fontSize)

    ax3.hist(repair_test_distn, density=True, alpha=0.2, label = 'Repair')
    ax3.hist(replace_test_distn, density=True, alpha=0.2, label = 'Replace')
    ax3.set_xlabel('URR Classifier Output',fontsize=fontSize)
    ax3.tick_params(axis='both', labelsize=fontSize)
    ax3.set_title('Test Set', fontsize=fontSize)

    plt.legend(loc="best",prop={'size':14},bbox_to_anchor=(1,0.5))
    plt.show()