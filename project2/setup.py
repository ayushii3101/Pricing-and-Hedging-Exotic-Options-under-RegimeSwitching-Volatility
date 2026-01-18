from setuptools import setup, find_packages

setup(
    name="regime-switching-options",
    version="1.0.0",
    description="Pricing and Hedging Exotic Options Under Regime-Switching Stochastic Volatility",
    author="Quantitative Finance Research Team",
    author_email="research@quantfinance.com",
    url="https://github.com/yourusername/regime-switching-options",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "numba>=0.54.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "plotly>=5.0.0",
        "pyyaml>=5.4.0",
        "tqdm>=4.62.0",
        "statsmodels>=0.13.0",
        "cvxpy>=1.1.0",
        "pydantic>=1.8.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "jupyter>=1.0.0",
            "ipywidgets>=7.6.0",
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
        ]
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "Topic :: Office/Business :: Financial",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    keywords="quantitative-finance options pricing hedging regime-switching stochastic-volatility",
)
