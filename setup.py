from setuptools import setup, find_packages

VERSION = "0.1.0"
DESCRIPTION = "Streamlit demos for carefree-learn"
with open("README.md", encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

setup(
    name="carefree-learn-apps",
    version=VERSION,
    packages=find_packages(exclude=("tests",)),
    install_requires=[
        "stqdm",
        "streamlit",
        "carefree-learn-deploy",
    ],
    author="carefree0910",
    author_email="syameimaru.saki@gmail.com",
    url="https://github.com/carefree0910/carefree-learn",
    download_url=f"https://github.com/carefree0910/carefree-learn-apps/archive/v{VERSION}.tar.gz",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    keywords="python streamlit PyTorch",
)
