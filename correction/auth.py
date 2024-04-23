from azure.identity import AzureCliCredential
from azure.core.credentials import AccessToken


def get_cognitiveservices_access_token(timeout: int = 100) -> str:
    credential = AzureCliCredential(process_timeout=timeout)
    token: AccessToken = credential.get_token("https://cognitiveservices.azure.com/.default")
    return token.token
