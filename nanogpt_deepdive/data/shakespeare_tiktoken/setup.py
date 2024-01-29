from setuptools import setup, find_namespace_packages

setup(
    name="shakespeare_char",
    packages=find_namespace_packages(include=["tiktoken_ext*"]),
    install_requires=["tiktoken"],
)
