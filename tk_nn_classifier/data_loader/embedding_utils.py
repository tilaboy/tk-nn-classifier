'''utils functions to read and save data'''
import os
import io
import shutil
import urllib
import urllib.request
import hashlib

from .. import LOGGER

DEFAULT_LANGUAGE = "en"
KNOWN_LANGUAGES = ("en", "de")

NEXUS_URL = "http://registry-01.p.nl01.textkernel.net:8081/repository/du-raw-repo"
TKML_GIT_REPO_URL = "http://tkgit-01.textkernel.net/textractor-team/tk-multilingual/raw/master"
TKVAC_GIT_REPO_URL = "http://tkgit-01.textkernel.net/textractor-team/vacancyparsing/raw/master"

EMBEDDINGS_FILENAMES = {
    'en': [TKML_GIT_REPO_URL, "en-wiki-and-cv-data-till-2016.bin"],
    'fr': [TKML_GIT_REPO_URL, "fr-cv.bin"],
    'de': [TKML_GIT_REPO_URL, "de-cv.bin"],
    'nl': [TKML_GIT_REPO_URL, "nl_wv-50-sw-100000-50.bin"],
    'es': [TKML_GIT_REPO_URL, "es-cv_wv-15.bin"],
    'it': [TKVAC_GIT_REPO_URL, "it-cvvacwiki.bin"]
}


def embeddings_url(lang: str) -> str:
    assert lang in EMBEDDINGS_FILENAMES and EMBEDDINGS_FILENAMES[lang], \
        "Embeddings not defined for language '{}'.".format(lang)
    repo_url, file_name = EMBEDDINGS_FILENAMES[lang]

    if repo_url == TKML_GIT_REPO_URL:
        embedding_url = "{}/CV-{}-ent/models/embeddings/{}".format(
            repo_url, lang.upper(), file_name)
    elif repo_url == TKVAC_GIT_REPO_URL:
        embedding_url = "{}/models/{}/extraction/models/embeddings/{}".format(
            repo_url, lang, file_name)
    else:
        raise ValueError('unknown repo for embedding')

    return embedding_url

def validate_model_checksum(language: str, target_file: str) -> bool:
    """Validate checksum match between local file and external file"""

    if os.path.exists(target_file):
        # checksum on local file
        chunk_num_blocks = 8192
        local_checksum = hashlib.md5()
        with open(target_file, 'rb') as istream:
            for chunk in iter(lambda: istream.read(chunk_num_blocks), b''):
                local_checksum.update(chunk)
        local_checksum = local_checksum.hexdigest()

        # checksum on external file
        # each word-embedding file has an associated .checksum file
        checksum_url = "{}.checksum".format(embeddings_url(language))
        with urllib.request.urlopen(checksum_url) as response:
            valid_checksum = response.read().decode('utf-8').strip()

        if local_checksum == valid_checksum:
            return True

    return False


def download_tk_embedding(language: str, target_file: str) -> None:
    """Download the word-embeddings if not already present"""

    download_file = True
    if os.path.exists(target_file):
        # also check for the file's expected checksum in case of an aborted download
        if validate_model_checksum(language, target_file):
            download_file = False
            LOGGER.info("  word-embeddings model is present and valid: skip download")
        else:
            LOGGER.info("  word-embeddings model is present, but invalid: re-download")
    else:
        LOGGER.info("  word-embeddings model missing: download {} embedding".format(language))

    if download_file:
        os.makedirs(os.path.dirname(target_file), exist_ok=True)
        url = embeddings_url(language)

        # noinspection PyUnresolvedReferences
        with urllib.request.urlopen(url) as response, open(target_file, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
        LOGGER.info("  downloaded the word-embeddings model")

        if not validate_model_checksum(language, target_file):
            raise ValueError(
                "The checksum of the downloaded model from '{0}' "
                "does not match the expected value from '{0}.checksum'".format(url))
