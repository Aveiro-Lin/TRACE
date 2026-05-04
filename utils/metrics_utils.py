import torch
from sklearn.metrics import fbeta_score, roc_auc_score, roc_curve, roc_curve, auc

def multi_accuracy(output, target):
    """computs the numti-label accuracy"""
    assert len(target.size()) == 2
    with torch.no_grad():
        batch_size, num_classes = target.size()
        
        # get set of labels 
        predict = torch.where(output > 0.5, torch.full_like(output, 1), torch.full_like(output, 0))
        
        # label_dict = {0:'AF', 1:'I-AVB', 2:'LBBB', 3:'RBBB', 4:'PAC', 5:'PVC', 6:'PAC', 7:'STD', 8:'STE'}
        TP = torch.zeros(num_classes)
        FP = torch.zeros(num_classes)
        TN = torch.zeros(num_classes)
        FN = torch.zeros(num_classes)
        #pdb.set_trace()
        Accuracy = torch.zeros(num_classes)
        Precision = torch.zeros(num_classes) 
        Recall =  torch.zeros(num_classes)
        F1 =  torch.zeros(num_classes)
        Specificity =  torch.zeros(num_classes)
        for label in range(num_classes):
            for sample in range(batch_size):
                if predict[sample, label] == target[sample, label] == 1:
                    TP[label] += 1
                if predict[sample, label] == 1 and target[sample, label] == 0:
                    FP[label] += 1
                if predict[sample, label] ==0 and target[sample, label] == 1:
                    FN[label] += 1
                if predict[sample, label] == target[sample, label] == 0:
                    TN[label] += 1
        epsilon = 1e-3
        for i in range(num_classes):
            Accuracy[i] = (TP[i] + TN[i]) / (TP[i] + FP[i] + TN[i] + FN[i] + epsilon) 
            Precision[i] = TP[i] / (TP[i] + FP[i] + epsilon) 
            Recall[i] = TP[i] / (TP[i] + FN[i] + epsilon) 
            F1[i] = 2 * (Precision[i] * Recall[i]) / (Precision[i] + Recall[i] + epsilon)    # 2*(Precision*Recall)/(Precision+Recall)
            Specificity[i] = TN[i] / (TN[i] + FP[i] + epsilon) 
        Accuracy_Macro = Accuracy.float().sum(0, keepdim=True).mul_(1.0 / num_classes)
        Precision_Macro = Precision.float().sum(0, keepdim=True).mul_(1.0 / num_classes)
        Recall_Macro = Recall.float().sum(0, keepdim=True).mul_(1.0 / num_classes)
        F1_Macro = F1.float().sum(0, keepdim=True).mul_(1.0 / num_classes)
        Specificity_Macro = Specificity.float().sum(0, keepdim=True).mul_(1.0 / num_classes)

        AUROC_Macro = [roc_auc_score(t, o, average='macro') for t, o in zip(target, output)]
        AUROC_Macro = sum(AUROC_Macro) / len(AUROC_Macro)
        return Accuracy_Macro, Precision_Macro, Recall_Macro, F1_Macro, AUROC_Macro, Specificity_Macro