def pytest_configure(config):
    config.addinivalue_line("markers", "slow: test loads a real model (deselect with -m 'not slow')")
