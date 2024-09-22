from setuptools import setup, find_packages

setup(
    name='mobile-recommendation-system',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'streamlit',
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'requests',
        'Flask',
        # Add more dependencies if needed
    ],
)
