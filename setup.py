import setuptools


def parse_requirements(filename):
    """Load requirements from a pip requirements file."""
    with open(filename) as f:
        reqs = (req.strip() for req in f.readlines() if req and not req.startswith("#"))
    return list(reqs)


install_requires = parse_requirements("requirements.txt")

setuptools.setup(
    name="ucl_dspp_dissertation",
    version="0.1",
    author="Lorenzo Gorini",
    author_email="lorenzo.gorini.22@ucl.ac.uk",
    description="""Dissertation Research Project 'The impact of Climate Change on
    italian tourism' for the 'MSc Data Science for Public Policy (Economics)' at UCL""",
    url="https://github.com/lorenz-gorini/ucl_dspp_dissertation",
    license="MIT",
    install_requires=install_requires,
    packages=["src"],
    zip_safe=False,
)
