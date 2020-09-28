from setuptools import setup, find_packages
import versioneer

setup(
    name="metagraph",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Graph algorithm solver across multiple hardware backends",
    author="Anaconda, Inc.",
    packages=find_packages(include=["metagraph", "metagraph.*"]),
    python_requires=">=3.7",
    install_requires=[
        "importlib_metadata",
        "numpy",
        "scipy",
        "donfig",
        "networkx",
        "pandas",
        "python-louvain",
        "nest-asyncio",
        "dask[array,dataframe,delayed]",
        "graphviz",
    ],
    include_package_data=True,
    entry_points={"metagraph.plugins": ["plugins=metagraph.plugins:find_plugins",]},
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
)
