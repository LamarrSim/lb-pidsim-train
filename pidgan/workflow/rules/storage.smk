import snakemake_storage_helper 

include: snakemake_storage_helper.__file__

class CustomS3(S3StorageProvider):
    @property 
    def _connector(self):
        return "s3://%(bucket)s" + f"/{config.get('storage_folder', '')}/".replace('//', '/')

    @_connector.setter
    def _connector(self, new_value):
        pass


storage s3data:
    **S3StorageProvider.from_secrets('s3data', config.get('secrets_file'))

storage s3images:
    **S3StorageProvider.from_secrets('s3images', config.get('secrets_file'))

storage s3models:
    **CustomS3.from_secrets('s3models', config.get('secrets_file'))

storage s3reports:
    **CustomS3.from_secrets('s3reports', config.get('secrets_file'))

    
def get_presigned_url(filenames, provider):
    import boto3, yaml

    if isinstance(filenames, str):
        filenames = [filenames]

    with open(config.get('secrets_file')) as sf:
        secrets = yaml.safe_load(sf)

    cfg = secrets[provider]
        
    s3_client = boto3.client(
        's3', 
        endpoint_url=cfg['endpoint_url'],
        aws_access_key_id=cfg['access_key'],
        aws_secret_access_key=cfg['secret_key'],
    )
    
    return [
        s3_client.generate_presigned_url(
            'get_object', 
            Params={'Bucket': cfg['bucket'], 'Key': filename},
            ExpiresIn=10000
        )
        for filename in filenames
    ]

    



