import torch
import pytest
import pygelib_cpp as backend


@pytest.fixture(scope='session', autouse=True)
def setup_GElibSession():
    backend.GElibSession()
    print("Setup GElibSession")
