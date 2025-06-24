from setuptools import setup, find_packages

setup(
    name="rakuten_mlops",
    version="0.1",
    packages=find_packages(include=["models", "models.*"]),
)