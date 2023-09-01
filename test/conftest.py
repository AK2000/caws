import pytest

def pytest_addoption(parser):
    parser.addoption(
        '--endpoint_id',
        action='store',
        required=True,
        help="Funcx compute id to run tests with"
    )

@pytest.fixture
def endpoint_id(request):
    return request.config.getoption("--endpoint_id")