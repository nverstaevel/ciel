from setuptools import setup

__version__ = "0.1.0"

metadata = dict(
    name="ciel",
    version=__version__,
    description="CIEL (Contextual Interactive Ensemble Learning)",
    author="Nicolas Vestaevel et al.",
    author_email="nicolas.verstaevel@irit.fr",
    license="MIT",
    packages=[
        "ciel",
        "ciel.torch_mas",
    ],
    install_requires=[
        "packaging",
        "scikit-learn",
        "torch",
        "scipy",
    ],
    python_requires=">=3.9",
    zip_safe=False,
    url="https://github.com/nverstaevel/ciel",  # use the URL to the github repo
    download_url="https://github.com/nverstaevel/ciel/releases",
)

setup(**metadata)
