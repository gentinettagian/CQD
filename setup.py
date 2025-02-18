from setuptools import setup, find_packages

VERSION = "0.0.2"
DESCRIPTION = "CQD"
LONG_DESCRIPTION = "CQD: Correcting and extending Trotterized quantum dynamics"

# Setting up
setup(
    name="cqd",
    version=VERSION,
    author="Gian Gentinetta",
    author_email="<gian.gentinetta@gmx.ch>",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=[
        "netket==3.14",
        "jax==0.4.35",
        "flax==0.8.3",
        "pennylane",
    ],  # add any additional packages that
    # needs to be installed along with your package. Eg: 'caer'
    keywords=["python", "hybrid dynamics"],
    classifiers=[],
)
