To install all the basic packages, run:
----------------------------------------

pip install -r requirements.txt

=========================================================================================================================
``PT_VQC-Tomography`` - Generate attacks based on state tomography, process tomography and non-unitary process tomography
=========================================================================================================================


.. image:: https://dl.circleci.com/status-badge/img/circleci/5ZWV663xqw4uDT8KDmJgpW/G4piVvQ66XDUHGX4Az1BJj/tree/circleci-project-setup.svg?style=shield&circle-token=41de148cb83684dd3c53509e74c3048071434118
        :target: https://dl.circleci.com/status-badge/redirect/circleci/5ZWV663xqw4uDT8KDmJgpW/G4piVvQ66XDUHGX4Az1BJj/tree/circleci-project-setup


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


Contents of requirements.txt

::

        matplotlib==3.5.2
        numpy==1.23.0
        pandas==1.4.3
        PennyLane==0.31.0
        qiskit==0.39.4
        qiskit_aer==0.11.2
        qiskit_ibmq_provider==0.19.2
        qiskit_ignis==0.7.1
        qiskit_terra==0.22.3
        qutip==4.7.1
        scikit_learn==1.1.1
        scipy==1.8.1
        sympy==1.12
        torch==1.12.0
        tqdm==4.64.1

What is PT_VQC?
-------------------

This repository combines all the codes to produce the plots and results from the following article: -----
A classical DNN attack has also been included, with the objective of recreating the initial state vector of each user based on the post-measurement classical output in QEPUF.
Additionally, the repository was complemented with the 1 qubit code for state emulation from the work of arXiv:1606.02734.
