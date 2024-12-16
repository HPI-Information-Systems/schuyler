from setuptools import setup

if __name__ == "__main__":
    setup(
        name="schuyler",
        version="0.0.1",
        description="",
        author="Lukas Laskowski",
        author_email="lukas.laskowski@hpi.de",
        url="https://github.com/HPI-Information-Systems/NumbER",
        license="MIT",
        classifiers=[
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
        ],
        # packages=find_packages(exclude=("tests", "tests.*")),
        packages=["schuyler"],
        package_data={"schuyler": ["py.typed"]},
        # install_requires=load_dependencies(),
        python_requires=">=3.7, <=3.11",
        # test_suite="tests",
        # cmdclass={
        #     "test": PyTestCommand,
        #     "typecheck": MyPyCheckCommand,
        #     "clean": CleanCommand,
        # },
        zip_safe=False,
    )