import unittest
import mlflow
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle

class TestModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Set up DagsHub credentials for MLflow tracking
        dagshub_token = os.getenv("DAGSHUB_PAT")
        if not dagshub_token:
            raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        dagshub_url = "https://dagshub.com"
        repo_owner = "campusx-official"
        repo_name = "mlops-project-2"

        # Set up MLflow tracking URI
        mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

        # Load the new model from MLflow model registry
        cls.new_model_name = "my_model"
        cls.new_model_version = cls.get_latest_model_version(cls.new_model_name)
        cls.new_model_uri = f'models:/{cls.new_model_name}/{cls.new_model_version}'
        cls.new_model = mlflow.pyfunc.load_model(cls.new_model_uri)

        # Load the vectorizer
        cls.vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))

        # Load holdout test data
        cls.holdout_data = pd.read_csv('data/processed/test_bow.csv')

    @staticmethod
    def get_latest_model_version(model_name, stage="Staging"):
        client = mlflow.MlflowClient()
        latest_version = client.get_latest_versions(model_name, stages=[stage])
        return latest_version[0].version if latest_version else None

    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.new_model)


    def test_model_signature(self):
        # Create a dummy input for the model based on expected input shape
        input_text = "hi how are you"
        
        # Transform input using vectorizer
        input_data = self.vectorizer.transform([input_text]).toarray()
        
        # Ensure input is in the expected format (e.g., numpy array)
        # If necessary, convert input_data to a DataFrame or a numpy array depending on the model input
        input_df = pd.DataFrame(input_data, columns=[str(i) for i in range(input_data.shape[1])])

        # Verify the input shape matches expected feature count
        self.assertEqual(input_df.shape[1], len(self.vectorizer.get_feature_names_out()))

        # Load model if needed (ensure you're using the correct method)
        model = mlflow.pyfunc.load_model(self.model_path)  # Ensure correct path
        
        # Predict using the model to verify input and output shapes
        prediction = model.predict(input_df)  # Ensure input is in the correct format

        # Verify the output shape
        self.assertEqual(len(prediction), input_df.shape[0])
        self.assertEqual(len(prediction.shape), 1)  # Assuming binary classification

if __name__ == "__main__":
    unittest.main()