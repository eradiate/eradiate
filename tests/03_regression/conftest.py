def pytest_itemcollected(item):
    item.add_marker("regression")
