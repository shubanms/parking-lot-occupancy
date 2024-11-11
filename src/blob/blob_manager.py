import os
import tempfile
from dotenv import load_dotenv
from ultralytics import YOLO
from azure.storage.blob import BlobServiceClient

load_dotenv()

models = {
    'v5': 'best_v5.pt',
    'v8': 'best_v8.pt',
    'v10': 'best_v10.pt',
    'v11': 'best_v11.pt'
}

class BlobManager:
    def __init__(self):
        self.account_name = os.environ.get("AZURE_STORAGE_ACCOUNT_NAME")
        self.access_key = os.environ.get("AZURE_STORAGE_ACCESS_KEY")
        self.container_name = os.environ.get("AZURE_STORAGE_CONTAINER_NAME")
        
        self.blob_service_client = BlobServiceClient(
            account_url=f"https://{self.account_name}.blob.core.windows.net",
            credential=self.access_key
        )
        
        self.loaded_models = {}  # Dictionary to store loaded models
        self._initialize_models()

    def _initialize_models(self):
        for version, model_filename in models.items():
            blob_client = self.blob_service_client.get_blob_client(container=self.container_name, blob=model_filename)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as temp_model_file:
                temp_model_file.write(blob_client.download_blob().readall())
                temp_model_path = temp_model_file.name

            try:
                self.loaded_models[version] = YOLO(temp_model_path)
                print(f"Loaded {version} model successfully.")
            except TypeError as e:
                print(f"Error loading model {version}: {e}")

    def get_model(self, model_version: str):
        return self.loaded_models.get(model_version)

    def upload_image(self, image_data, image_name: str):
        blob_client = self.blob_service_client.get_blob_client(container=self.container_name, blob=image_name)
        blob_client.upload_blob(image_data, overwrite=True)

    def load_image(self, image_name):
        blob_client = self.blob_service_client.get_blob_client(container=self.container_name, blob=image_name)
        return blob_client.download_blob().readall()

blob_manager = BlobManager()
