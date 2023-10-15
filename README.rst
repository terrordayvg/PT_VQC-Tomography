# PT_VQC-Tomography
This repository combines all the codes to produce the plots and results from the following article: -----
A classical DNN attack has also been included, with the objective of recreating the initial state vector of each user based on the post-measurement classical output in QEPUF.
Additionally, the repository was complemented with the 1 qubit code for state emulation from the work of arXiv:1606.02734.

To install all the basic packages, run:

pip install -r requirements.txt

=========================================================================================================================
``PT_VQC-Tomography`` - Generate attacks based on state tomography, process tomography and non-unitary process tomography
=========================================================================================================================


.. image:: https://circleci.com/gh/circleci/PT_VQC-Tomography.svg?style=svg
    :target: https://circleci.com/gh/circleci/PT_VQC-Tomography


.. image:: https://img.shields.io/travis/bndr/pipreqs.svg
        :target: https://travis-ci.org/bndr/pipreqs


.. image:: https://img.shields.io/pypi/v/pipreqs.svg
        :target: https://pypi.python.org/pypi/pipreqs


.. image:: https://codecov.io/gh/bndr/pipreqs/branch/master/graph/badge.svg?token=0rfPfUZEAX
        :target: https://codecov.io/gh/bndr/pipreqs

.. image:: https://img.shields.io/pypi/l/pipreqs.svg
        :target: https://pypi.python.org/pypi/pipreqs



Installation of required libraries
------------

::

    install -r requirements.txt

Usage
-----

::

    Usage:
        pipreqs [options] [<path>]

    Arguments:
        <path>                The path to the directory containing the application files for which a requirements file
                              should be generated (defaults to the current working directory)

    Options:
        --use-local           Use ONLY local package info instead of querying PyPI
        --pypi-server <url>   Use custom PyPi server
        --proxy <url>         Use Proxy, parameter will be passed to requests library. You can also just set the
                              environments parameter in your terminal:
                              $ export HTTP_PROXY="http://10.10.1.10:3128"
                              $ export HTTPS_PROXY="https://10.10.1.10:1080"
        --debug               Print debug information
        --ignore <dirs>...    Ignore extra directories, each separated by a comma
        --no-follow-links     Do not follow symbolic links in the project
        --encoding <charset>  Use encoding parameter for file open
        --savepath <file>     Save the list of requirements in the given file
        --print               Output the list of requirements in the standard output
        --force               Overwrite existing requirements.txt
        --diff <file>         Compare modules in requirements.txt to project imports
        --clean <file>        Clean up requirements.txt by removing modules that are not imported in project
        --mode <scheme>       Enables dynamic versioning with <compat>, <gt> or <non-pin> schemes
                              <compat> | e.g. Flask~=1.1.2
                              <gt>     | e.g. Flask>=1.1.2
                              <no-pin> | e.g. Flask

Example
-------

::

    $ pipreqs /home/project/location
    Successfully saved requirements file in /home/project/location/requirements.txt

Contents of requirements.txt

::

    wheel==0.23.0
    Yarg==0.1.9
    docopt==0.6.2

Why not pip freeze?
-------------------

- ``pip freeze`` only saves the packages that are installed with ``pip install`` in your environment.
- ``pip freeze`` saves all packages in the environment including those that you don't use in your current project (if you don't have ``virtualenv``).
- and sometimes you just need to create ``requirements.txt`` for a new project without installing modules.
