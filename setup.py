from setuptools import setup, find_packages

setup(
    name="sro_sto_plume",
    version="0.1",
    description="Repository for SRO_STO_Plume project",
    author="Yichen Guo",
    package_dir={"": "src"},
    packages=find_packages(where="src"),  # Will include all packages under 'src'
    install_requires=[],  # Add dependencies if needed, e.g., ['numpy']
)
