import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso
from TaxiFareModel.encoders import TimeFeaturesEncoder
from TaxiFareModel.encoders import DistanceTransformer

class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        distance_transformer = Pipeline([
            ('distance_transformer', DistanceTransformer()),
            ('standard_scaling', StandardScaler())
            ])

        time_feat_encoder = Pipeline([
            ('time_encoder', TimeFeaturesEncoder('pickup_datetime')),
            ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore'))
            ])

        preprocess = ColumnTransformer([
            ('distance_transformer', distance_transformer, ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']),
            ('time_encoder', time_feat_encoder, ['pickup_datetime'])],
            remainder='drop'
            )

        final_pipe = Pipeline([
            ('preprocessing', preprocess),
            ('regressor', Lasso())
            ])

        return final_pipe

    def run(self):
        """set and train the pipeline"""

        return self.set_pipeline().fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""

        y_pred = self.run().predict(X_test)

        return np.sqrt(((y_pred - y_test)**2).mean())


if __name__ == "__main__":
    # get data
    # clean data
    # set X and y
    # hold out
    # train
    # evaluate
    print('TODO')
