from setuptools import setup, find_packages

setup(
    name="summit",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langgraph",
        "langgraph-prebuilt", 
        "langgraph-sdk",
        "langgraph-checkpoint-sqlite",
        "langsmith",
        "langchain-community",
        "langchain-core",
        "langchain-openai",
        "trustcall",
        "langgraph-cli[inmem]",
        "psycopg2-binary",
    ],
    python_requires=">=3.8",
)