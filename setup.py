from setuptools import setup, find_packages

setup(
    name="rakuten_ds",
    version="0.1",
    packages=find_packages(include=["app", "app.*"]),
)