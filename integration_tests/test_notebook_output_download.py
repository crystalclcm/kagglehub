import unittest

import pytest
from requests import HTTPError

from kagglehub import notebook_output_download

from .utils import assert_files, create_test_cache, unauthenticated


class TestModelDownload(unittest.TestCase):
    def test_download_notebook_output_succeeds(self) -> None:
        with create_test_cache():
            actual_path = notebook_output_download("alexisbcook/titanic-tutorial")

            expected_files = ["submission.csv"]
            assert_files(self, actual_path, expected_files)

    @pytest.mark.skip(reason="TODO(b/380948485): support anonymous access to `/api/v1/kernels`")
    def test_download_public_notebook_output_as_unauthenticated_succeeds(self) -> None:
        with create_test_cache():
            with unauthenticated():
                actual_path = notebook_output_download("alexisbcook/titanic-tutorial")

                expected_files = ["submission.csv"]
                assert_files(self, actual_path, expected_files)

    def test_download_private_notebook_output_succeeds(self) -> None:
        with create_test_cache():
            actual_path = notebook_output_download("integrationtester/private-titanic-tutorial")

            expected_files = ["submission-01.csv", "submission-02.csv"]

            assert_files(self, actual_path, expected_files)

    def test_download_private_notebook_output_single_file_succeeds(self) -> None:
        with create_test_cache():
            actual_path = notebook_output_download(
                "integrationtester/private-titanic-tutorial", path="submission-02.csv"
            )

            expected_files = ["submission-02.csv"]

            assert_files(self, actual_path, expected_files)

    def test_download_large_notebook_output_warns(self) -> None:
        handle = "integrationtester/titanic-tutorial-many-output-files"
        with create_test_cache():
            msg = f"Too many files in {handle} (capped at 25). Unable to download files."
            # If the model has > 25 files, we warn the user that it's not supported yet
            with self.assertWarns(
                Warning,
                msg=msg,
            ):
                notebook_output_download(handle)

    def test_download_private_notebook_output_with_incorrect_file_path_fails(self) -> None:
        with create_test_cache(), self.assertRaises(HTTPError):
            notebook_output_download("integrationtester/titanic-tutorial", path="submission-03.csv")
