from setuptools import setup, find_packages

setup(
    name="cgmml",
    version="3.0.0-alpha",
    author="Markus Hinsche",
    author_email="mhinsche@childgrowthmonitor.com",
    description="cgm-ml",
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Welthungerhilfe/cgm-ml",
    classifiers=[
        'Intended Audience :: Healthcare Industry',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GPL-3.0',
        'Operating System :: OS Independent',
    ],
    # package_dir={"": "src"},
    packages=find_packages(
        include="src/common/model_utils",
        exclude=['docs', 'src/common/model_utils/tests'],
        ),
    python_requires=">=3.6",
)