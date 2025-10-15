import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score, confusion_matrix, make_scorer,roc_auc_score

# models is a dictionary where the keys correspond to the names of specific models, 
# and the values correspond to the initialization results of the models corresponding to those names.

def model_build(models, X_train, X_test, y_train, y_test):

    def specificity(y_true, y_pred):
        try:
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            return tn / (tn + fp) if (tn + fp) > 0 else 0.0
        except:
            return 0.0
        
    specificity_scorer = make_scorer(specificity)
    train_results = []
    test_results = []
    scoring = {
    'f1_macro': 'f1_macro',
    'precision_macro': 'precision_macro',
    'recall_macro': 'recall_macro',
    'accuracy': 'accuracy',
    'roc_auc': 'roc_auc',
    'specificity': specificity_scorer  
        }
    
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    for name, model in models.items():
        print(f'\n{"="*10}\nEvaluating {name}\n{"="*10}')
        cv_results = cross_validate(
            model, X_train, y_train,
            cv=cv, scoring=scoring, n_jobs=1, return_train_score=False
        )
        

        train_metrics = {
            'Model': name,
            'F1_macro': np.mean(cv_results['test_f1_macro']),
            'Precision_macro': np.mean(cv_results['test_precision_macro']),
            'Recall_macro': np.mean(cv_results['test_recall_macro']),
            'Accuracy': np.mean(cv_results['test_accuracy']),
            'ROC_AUC': np.mean(cv_results['test_roc_auc']),
            'Specificity': np.mean(cv_results['test_specificity'])
        }

        train_results.append(train_metrics)
        

    
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        test_metrics = {
            'Model': name,
            'F1_macro': f1_score(y_test, y_pred, average='macro'),
            'Precision_macro': precision_score(y_test, y_pred, average='macro'),
            'Recall_macro': recall_score(y_test, y_pred, average='macro'),
            'Accuracy': accuracy_score(y_test, y_pred),
            'ROC_AUC': roc_auc_score(y_test, y_proba),
            'Specificity': specificity(y_test, y_pred)
        }
        test_results.append(test_metrics)


    train_df = pd.DataFrame(train_results).set_index('Model')
    test_df = pd.DataFrame(test_results).set_index('Model')
    return train_df,test_df