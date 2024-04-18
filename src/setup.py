from setuptools import setup, find_packages

setup(
    name="FinancialModels",
    version="0.0.1",
    author="Tariq Ouahmane",
    author_email="ta.ouahmane@gmail.com",
    #url="https://www.youtube.com/channel/UCv9MUffHWyo2GgLIDLVu0KQ",
    description="Mathematical models used in finance",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[],
    entry_points={"console_scripts": ["FinancialModels = src.main:main"]},
)