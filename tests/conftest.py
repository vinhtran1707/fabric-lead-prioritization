import pytest
from leadscore.lakehouse import get_spark


@pytest.fixture(scope="session")
def spark():
    s = get_spark("leadscore-tests")
    s.sparkContext.setLogLevel("ERROR")
    yield s
