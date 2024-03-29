import os
import sys

from src.exception import CustomException


class S3Operation:
    def sync_folder_to_s3(
        self, folder: str, bucket_name: str, bucket_folder_name: str
    ) -> None:
        try:
            os.system(f"aws s3 sync {folder} s3://{bucket_name}/{bucket_folder_name}/ ")

        except Exception as e:
            raise CustomException(e, sys)
