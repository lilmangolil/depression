from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd

# X_train is the training set divided from X
# X_test is the test set divided from X
# numeric_features is a list composed of continuous variable names in order
# categorical_features is a list composed of categorical variable names in order

def data_preprocessing(X_train,X_test,numeric_features,categorical_features):

    preprocessor = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', StandardScaler(), numeric_features)            
        ])

    X_train_processed = preprocessor.fit_transform(pd.DataFrame(X_train))
    X_test_processed = preprocessor.transform(pd.DataFrame(X_test))

    print("the shape of X_train_processed: ", X_train_processed.shape)
    print("the shape of X_test_processed:", X_test_processed.shape)
    return X_train_processed,X_test_processed