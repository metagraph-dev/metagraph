from setuptools import setup, find_packages
import versioneer

setup(
    name="metagraph",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Graph algorithm solver across multiple hardware backends",
    author="Anaconda, Inc.",
    packages=find_packages(include=["metagraph", "metagraph.*"]),
    install_requires=["importlib_metadata", "numpy"],
    include_package_data=True,
    entry_points={
        "metagraph.plugins": "plugins=metagraph.default_plugins:find_plugins"
    },
)
