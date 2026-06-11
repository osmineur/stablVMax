import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='Stabl',
    version='1.0.0',
    author='Grégoire Bellan',
    author_email='gbellan@surge.care',
    description='Stabl light weight',
    packages=['stabl'],
    install_requires=[
        'joblib>=1.4.2',
        'tqdm>=4.66.1',
        'numpy>=1.26.4',
        'matplotlib>=3.8.2',
        "knockpy>=1.3.1",
        "scikit-learn>=1.4.2",
        "seaborn>=0.13.0",
        "pandas>=2.2.2",
        "statsmodels>=0.14.5",
        "openpyxl>=3.1.2",
        "adjustText>=1.1.1",
        "scipy>=1.11.4",
        "osqp>=0.6.3",
        "xgboost>=3.0.5",
        "unionfind>=0.0.22",
    ]
)
