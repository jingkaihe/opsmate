# coding: utf-8

"""
    FastAPI

    No description provided (generated by Openapi Generator https://github.com/openapitools/openapi-generator)

    The version of the OpenAPI document: 0.1.0
    Generated by OpenAPI Generator (https://openapi-generator.tech)

    Do not edit the class manually.
"""  # noqa: E501


import unittest

from api.default_api import DefaultApi


class TestDefaultApi(unittest.TestCase):
    """DefaultApi unit test stubs"""

    def setUp(self) -> None:
        self.api = DefaultApi()

    def tearDown(self) -> None:
        pass

    def test_health_v1_health_get(self) -> None:
        """Test case for health_v1_health_get

        Health
        """
        pass

    def test_models_v1_models_get(self) -> None:
        """Test case for models_v1_models_get

        Models
        """
        pass

    def test_run_v1_run_post(self) -> None:
        """Test case for run_v1_run_post

        Run
        """
        pass


if __name__ == '__main__':
    unittest.main()
