import logging
from typing import Optional

from kagglehub import registry
from kagglehub.gcs_upload import upload_files
from kagglehub.handle import parse_dataset_handle

logger = logging.getLogger(__name__)

def dataset_download(handle: str, path: Optional[str] = None, *, force_download: Optional[bool] = False) -> str:
    """Download dataset files
    Args:
        handle: (string) the dataset handle
        path: (string) Optional path to a file within the dataset
        force_download: (bool) Optional flag to force download a dataset, even if it's cached.
    Returns:
        A string requesting the path to the requested dataset files.
    """

    h = parse_dataset_handle(handle)
    return registry.dataset_resolver(h, path, force_download=force_download)