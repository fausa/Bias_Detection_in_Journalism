# Import basic libraries
import numpy as np
import pandas as pd

# Import visualization libraries
import matplotlib.pyplot as plt

# Import model and performance evaluation libraries
from sklearn.base import is_classifier, is_regressor

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import KNNImputer, SimpleImputer

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer

from sklearn.metrics import accuracy_score, r2_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

# Import balancing library
from imblearn.over_sampling import SMOTE
#from imblearn.over_sampling import RandomOverSampler

# Import utility libraries
from time import time

"""
This class attempts to encapsulate 'basic' scikit-learn modeling
capabilities - including pre-model, model, and evaluation - into a
common model process code base, to leverage tested commonality and
enable focus on model inputs and outputs, vs. mechanics.
"""
class ModelProcess:
    show_progress = False

    def __init__(self,
                 algorithm=None,
                 id=None,
                 params=None,
                 X_train=None, y_train=None,
                 X_val=None, y_val=None,
                 X_test=None, y_test=None,
                 classes=None,
                 cat_cols=None, num_cols=None,
                 balance_target=False):
        self.algorithm = algorithm
        self.id = id
        self.params = params
        self.X = {'train': X_train, 'val': X_val, 'test': X_test}
        self.y = {'train': y_train, 'val': y_val, 'test': y_test}

        self.classes = classes if classes is not None else sorted(y_train.iloc[:, 0].unique())
        if len(self.classes) > 2:
            self.f1_average = 'macro'
        else:
            self.f1_average = 'binary'

        self.cat_cols = cat_cols
        self.num_cols = num_cols

        self.balance_target = balance_target

        # Set descriptive model process name
        self.name = self.algorithm.__class__.__name__
        if id is not None:
            self.name += ' (' + str(id) + ')'
            
        # Prepare algorithm with hyperparameters+
        if params is not None:
            prev_params = self.algorithm.get_params(deep=False)
            if (prev_params is not None) & (params is not None):
                params = {**prev_params, **params}
            self.algorithm.set_params(**params)

        # Create transformer to automatically encode non-numeric values
        #   (Note not necessarily the same as the Class parameter cat_cols,
        #   which may already be encoded and simply be specifying a need to
        #   impute using the 'categorical' imputation approach here)
        self.enc_cols = self.X['train'].select_dtypes(include=['object', 'category']).columns
        self.enc_transformer = make_pipeline(OneHotEncoder(handle_unknown='ignore',
                                                           sparse=False)) 

        # Create categorical transformer (for pipeline-based imputation)
        self.cat_transformer = make_pipeline(SimpleImputer(strategy='most_frequent',
                                                           fill_value='missing'))
        
        # Create numerical transformer (for both imputation and scaling)
        self.num_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())])

        # Create preprocessor using transformers created above
        self.transformers = []
# This code not needed for project; commenting-out for refinement
#        if self.cat_cols is not None:
#            self.transformers.append(('cat', self.cat_transformer,
#                                      self.X['train'][self.cat_cols].columns))
        if self.enc_cols is not None:
            self.transformers.append(('enc', self.enc_transformer,
                                      self.X['train'][self.enc_cols].columns))
        if self.num_cols is not None:
            self.transformers.append(('num', self.num_transformer,
                                      self.X['train'][self.num_cols].columns))
        if self.transformers != ():
            self.preprocessor = ColumnTransformer(list(self.transformers), remainder='passthrough')
        else:
            self.preprocessor = None

        # Create base pipeline
        self.pipe = Pipeline(steps=[('preprocessor', self.preprocessor),
                                    ('model', self.algorithm)])

        self.model = self.pipe.named_steps['model']

        # Create landing place for base predictions, scores, and run times
        self.pred = {'train': [None], 'val': None, 'test': None}
        self.pred_proba = {'train': None, 'val': None, 'test': None}
        self.score = {'train': [None, None], 'val': [None, None], 'test': [None, None]}
        self.time = {'train': None, 'val': None, 'test': None}

    def __run(self, dataset):
        if dataset not in self.X:
            raise ValueError('%r not recognized' % dataset)
        if (self.X[dataset] is None) | (self.y[dataset] is None):
            raise ValueError('missing data')

        if (self.show_progress):
            print(self.name + ': ' + dataset + '...', end=' ')
        start = time()

        if dataset == 'train':
            if self.balance_target:
                smote = SMOTE(random_state=42)
                self.preprocessor.fit(self.X[dataset])
                X_preprocessed = self.preprocessor.transform(self.X[dataset])
                X_resampled, y_resampled = smote.fit_resample(X_preprocessed, np.ravel(self.y[dataset]))
                self.model.fit(X_resampled, y_resampled)
            else:
                self.pipe.fit(self.X[dataset], np.ravel(self.y[dataset]))

        self.pred[dataset] = self.pipe.predict(self.X[dataset])
        if is_classifier(self.algorithm):
            if hasattr(self.model, 'predict_proba'):
                self.pred_proba[dataset] = self.pipe.predict_proba(self.X[dataset])

            # Temporarily, avoid 'crash' if e.g., y_true doesn't exist (may be case for 'test', for example)
            try:
                self.score[dataset] = [accuracy_score(self.y[dataset], self.pred[dataset]), 
                                    f1_score(self.y[dataset], self.pred[dataset], average=self.f1_average)]
            except:
                pass

        else:
            try:
                self.score[dataset] = r2_score(self.y[dataset], self.pred[dataset])
            except:
                pass

        self.time[dataset] = time() - start
        if (self.show_progress):
            print('done in %0.2fs.' % self.time[dataset])

        return self

    def train(self):
        self._ModelProcess__run('train')
        return self

    def validate(self):
        self._ModelProcess__run('val')
        return self
    
    def test(self):
        self._ModelProcess__run('test')
        
    def train_validate(self):
        self.train()
        self.validate()
        return self
    
    def train_validate_test(self):
        self.train_validate()
        self.test()
        return self
        
    def summary_df(self):
        return pd.DataFrame({
            'algorithm': self.name,
            'parameters': str(self.params),
            'train_acc': self.score['train'][0],
            'train_f1': self.score['train'][1],
            'val_acc': self.score['val'][0],
            'val_f1': self.score['val'][1],
            'test_acc': self.score['test'][0],
            'test_f1': self.score['test'][1]}, index=[0])
        
    def summary(self, dataset='train'):
        if dataset not in self.X:
            raise ValueError('%r not recognized' % dataset)

        print('\n\n' + self.name + ' - ' + dataset + ' dataset')
        if is_classifier(self.algorithm):
            try:
                print(); print(classification_report(self.y[dataset], self.pred[dataset], zero_division=0))
            except:
                pass
        else:
            if dataset != 'train':
                print('\n(Train R2 %0f)' % self.score['train'][1], end='')
            print('\n(R2 %0f)' % self.score[dataset][1])
#            try:
# removing regressionSummary() in favor of more scikit-focused                
#                print(regressionSummary(self.y[dataset], self.pred[dataset]))
#            except:
#                pass
            
    def confusion_matrix(self, dataset='train'):
        fig, ax = plt.subplots()
        try:
            cmd = ConfusionMatrixDisplay(confusion_matrix(self.y[dataset], self.pred[dataset]))
        except:
            pass
        cmd.plot(ax = ax)
        plt.suptitle('Confusion Matrix', y = 1)
        plt.title(self.name + ' - ' +  dataset + ' dataset')
        plt.show()

    def estimate(self):
        if hasattr(self.model, 'coef_'):
            est_df = pd.DataFrame({'Coeff': np.squeeze(self.model.coef_)},
                                  index=self.X['train'].columns)
            est_df.sort_values(by=['Coeff'], ascending=False, inplace=True)
            print('\n\n' + self.name)
            print('\nInterecept', np.squeeze(self.model.intercept_),
                  '\n', est_df.to_string())