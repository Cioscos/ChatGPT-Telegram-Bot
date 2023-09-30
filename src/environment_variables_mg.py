import logging
import os

logger = logging.getLogger(__name__)

KEYRING = {
    'Telegram': None,
    'DevId': None,
    'OpenAI': None
}


def keyring_initialize():
    """
    Initialize the keyring.
    This functions load from decrypted partition the keys needed.

    :return: True if everything went fine, False otherwise.
    """
    root_path = os.environ.get('KEYRING')
    if root_path is None:
        logger.error('Root directory is not set')
        return None

    # Get the telegram token key
    with open(os.path.join(root_path, 'telegram.dat')) as file:
        KEYRING['Telegram'] = file.read().strip()

    # Get OpenAI token key
    with open(os.path.join(root_path, 'openai.dat')) as file:
        KEYRING['OpenAI'] = file.read().strip()

    with open(os.path.join(root_path, 'dev_id.dat')) as file:
        KEYRING['DevId'] = file.read().strip()

    return True


def keyring_get(service):
    """
    Get the key related to a service from the keyring.

    :param service: The service which get the key.
    :return: The keyring in the keyring. None if it's not present.
    """
    if service not in KEYRING:
        logger.warning('Key of service %s not found in keyring' % service)
        return None

    return KEYRING[service]
