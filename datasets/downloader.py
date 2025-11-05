import boto3
from botocore import UNSIGNED
from botocore.client import Config


def get_file(filename, destination_filename, bucket_name='astra-vector', region='us-east-1'):
    s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED),
                             region_name=region)
    with open(destination_filename, 'wb') as f:
        s3_client.download_fileobj(bucket_name, f'wikipedia_squad/{filename}', f)
