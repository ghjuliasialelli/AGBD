"""

This file contains the Random Forest model class.

"""

#######################################################################################################################
# Imports

import lightgbm as lgb
import numpy as np

#######################################################################################################################
# Helper functions 

def rmse_func(preds, val_data):
    targets = val_data.get_label()
    eval_result = np.sqrt(np.mean((preds - targets) ** 2))
    return ('rmse', eval_result, False)

class RandomForest():
    
    def __init__(self, model_name, num_leaves = 165, max_depth = -1, learning_rate = 0.1, 
                 num_iterations = 1000, min_split_gain = 0., min_child_weight = 1e-3, 
                 min_child_samples = 20, reg_alpha = 0, reg_lambda = 5, bagging_fraction = 0.5,
                 bagging_freq = 1, n_jobs = 3):

        # Arguments for the model
        self.model_name = model_name
        self.params = {'max_leaves': num_leaves, 'max_depth': max_depth, 
                       'learning_rate': learning_rate, 'num_iterations': num_iterations, 
                       'min_split_gain': min_split_gain, 'min_child_weight': min_child_weight,
                       'min_child_samples': min_child_samples, 'reg_alpha': reg_alpha, 
                       'reg_lambda': reg_lambda, 'n_jobs': n_jobs, 'objective': 'regression',
                       'boosting_type': 'gbdt', 'bagging_fraction': bagging_fraction, 
                       'bagging_freq': bagging_freq, 'force_col_wise' : True, 'verbose': -1, 
                       'metric': 'rmse'}

        # For the ensemble
        self.val_rmses = []
        self.test_rmses = []

    def fit(self, train_dataset, val_dataset, test_dataset):
        """
        This function fits the model to the training set and evaluates it on the validation and test sets.
        """

        # Fit the models

        for idx in range(1, 6) :

            print(f'Fitting model {idx}...')
            self.params['seed'] = 3 + idx
            self.booster = lgb.train(params = self.params,
                                    train_set = lgb.Dataset(train_dataset.features, train_dataset.labels), 
                                    keep_training_booster = True,
                                    valid_sets = [lgb.Dataset(val_dataset.features, val_dataset.labels)],
                                    valid_names = ['val'],
                                    callbacks = [lgb.early_stopping(stopping_rounds = 50)],
                                    feval = rmse_func)
            
            # Get metrics on validation set
            val_preds = self.booster.predict(val_dataset.features)
            val_rmse = np.sqrt(np.mean(np.power(val_preds - val_dataset.labels['agbd'].to_numpy(), 2)))
            self.val_rmses.append(val_rmse)
            print('Validation RMSE :', val_rmse)

            # Get metrics on test set
            test_preds = self.booster.predict(test_dataset.features)
            test_rmse = np.sqrt(np.mean(np.power(test_preds - test_dataset.labels['agbd'].to_numpy(), 2)))
            self.test_rmses.append(test_rmse)
            print('Test RMSE :', test_rmse)

            # Plot and save the feature importance
            ax = lgb.plot_importance(self.booster, importance_type = 'gain', figsize = (10, 5))
            ax.figure.savefig(f'fimp/{self.model_name}-{idx}.png')

            # Save the model
            self.booster.save_model(f'weights/{self.model_name}-{idx}.txt')

        self.ens_val_rmse = np.mean(self.val_rmses)
        self.ens_val_std = np.std(self.val_rmses)
        self.ens_test_rmse = np.mean(self.test_rmses)
        self.ens_test_std = np.std(self.test_rmses)

        # And print the results
        print(f'Ensemble Validation RMSE : {self.ens_val_rmse} +/- {self.ens_val_std}')
        print(f'Ensemble Test RMSE : {self.ens_test_rmse} +/- {self.ens_test_std}')