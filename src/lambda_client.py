import boto3
import json
import os
from botocore.config import Config

class LambdaClient():
    def __init__(self):
        self.config = Config(
            retries={"max_attempts": 0},
            read_timeout=10000,
            connect_timeout=10000
        )
        self.session = boto3.Session()
        self.client = self.session.client(service_name='lambda', region_name='us-east-2', config=self.config)

    def invoke_lambda(self, payload):
        response = self.client.invoke(
            FunctionName=os.getenv("LAMBDA_ARN") or "",
            InvocationType="RequestResponse",
            Payload=json.dumps(payload)
        )
        result = response['Payload'].read()
        return json.loads(result)