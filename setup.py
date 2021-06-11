from setuptools import setup, find_packages

setup(
    name="cgm-ml-common",
    version="3.0.0-alpha5",
    author="Markus Hinsche",
    author_email="mhinsche@childgrowthmonitor.com",
    url="https://github.com/Welthungerhilfe/cgm-ml",

    description="ChildGrowthMonitor's ML Common code",
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",

    classifiers=[
        'Intended Audience :: Healthcare Industry',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
    ],

    packages=find_packages(),

    python_requires=">=3.6",
    setup_requires=['pytest-runner', 'flake8'],
    tests_require=['pytest'],
)
