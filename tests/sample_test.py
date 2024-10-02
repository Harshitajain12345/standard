import unittest
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        # Sample dataframe similar to what is shown in the screenshot
        self.df = pd.DataFrame({
            'age': [60, 27, 45],
            'gender': ['Male', 'Male', 'Female'],
            'fever': [103.0, 100.0, 102.0],
            'has_covid': ['Yes', 'No', 'Yes'],
            'cough': ['Yes', 'No', 'Yes'],
            'city': ['New York', 'Los Angeles', 'Chicago']
        })
        
        # Encoding columns 'gender', 'has_covid', 'cough', 'city'
        self.lb = LabelEncoder()
        self.df['gender'] = self.lb.fit_transform(self.df['gender'])
        self.df['has_covid'] = self.lb.fit_transform(self.df['has_covid'])
        self.df['cough'] = self.lb.fit_transform(self.df['cough'])
        self.df['city'] = self.lb.fit_transform(self.df['city'])
        
    def test_label_encoding(self):
        # Test if Label Encoding works correctly
        self.assertIn(0, self.df['gender'].values)
        self.assertIn(1, self.df['has_covid'].values)

    def test_train_test_split(self):
        # Dropping the target column 'has_covid'
        X = self.df.drop(columns=['has_covid'])
        y = self.df['has_covid']

        # Splitting the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Test the shapes of train/test split
        self.assertEqual(len(X_train), 2)
        self.assertEqual(len(X_test), 1)

    def test_minmax_scaling(self):
        # Dropping the target column 'has_covid'
        X = self.df.drop(columns=['has_covid'])
        y = self.df['has_covid']

        # Apply MinMaxScaler
        mm = MinMaxScaler()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_scaled = mm.fit_transform(X_train)

        # Test that scaling keeps values between 0 and 1
        self.assertTrue((X_train_scaled >= 0).all())
        self.assertTrue((X_train_scaled <= 1).all())

    def test_standard_scaling(self):
        # Dropping the target column 'has_covid'
        X = self.df.drop(columns=['has_covid'])
        y = self.df['has_covid']

        # Apply StandardScaler
        sc = StandardScaler()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_scaled = sc.fit_transform(X_train)

        # Test that scaled data has zero mean and unit variance
        np.testing.assert_almost_equal(X_train_scaled.mean(), 0, decimal=1)
        np.testing.assert_almost_equal(X_train_scaled.std(), 1, decimal=1)

if __name__ == '__main__':
    unittest.main()
