import pathlib
from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="temporal-taggers",
    version="0.0.1",
    description="Neural temporal taggers with Transformer architectures",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/satya77/Transformer_Temporal_Tagger",
    author="Satya Almasian and Dennis Aumiller",
    author_email="almasian@informatik.uni-heidelberg.de",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=find_packages(),
    include_package_data=True,
    install_requires=["transformers>=4.3.3", "torch>=1.8", "datasets", "spacy>=3.0", "beautifulsoup4>=4.9",
                      "seqeval", "conllu", "filelock"],
)
