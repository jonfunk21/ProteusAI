from setuptools import setup, find_packages

# Read the content of requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    author="Jonathan Funk",
    author_email="funk.jonathan21@gmail.com",
    description="ProteusAI is a python package designed for AI driven protein engineering.",
    url="https://github.com/jonfunk21/ProteusAI",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.8",
)
