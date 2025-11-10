=========================================================================================================================
``PT_VQC and U-VQSVD Tomography`` - Generate attacks based on state tomography, process tomography and non-unitary process tomography
=========================================================================================================================


.. image:: https://dl.circleci.com/status-badge/img/circleci/5ZWV663xqw4uDT8KDmJgpW/G4piVvQ66XDUHGX4Az1BJj/tree/circleci-project-setup.svg?style=shield&circle-token=41de148cb83684dd3c53509e74c3048071434118
        :target: https://dl.circleci.com/status-badge/redirect/circleci/5ZWV663xqw4uDT8KDmJgpW/G4piVvQ66XDUHGX4Az1BJj/tree/circleci-project-setup



.. image:: https://codecov.io/gh/terrordayvg/PT_VQC-Tomography/graph/badge.svg?token=880RTY0T96
        :target: https://codecov.io/gh/terrordayvg/PT_VQC-Tomography

.. image:: https://img.shields.io/badge/python-3.11-blue.svg
        :target: https://www.python.org/downloads/release/python-3110/


Installation of required libraries

::

    install -r requirements.txt


Usage

               PT_VQC is divided into 3 main topics, responsible for:
        
        Folders:  
                * `State tomography`.
                * `Process tomography`.
                * `Non-unitary process tomography`

                
        Aditional: 
                * `Classical Deep Neural Network (DNN) attack for QEPUF initialization reconstruction (Classical-DNN-PUF-attack)`.

        Tests:  
                * Pytest in Test folder, for all major functions in the codes.
                * CircleCI is integrated for continuous integration (.circleci folder).


Contents of requirements.txt
-----

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
        pytest==7.4.2

How to Cite?
===========

If you use this work, please cite the following paper:

::

    @article{Galetsky_2024,
      doi = {10.1088/1367-2630/ad5df1},
      url = {https://doi.org/10.1088/1367-2630/ad5df1},
      year = {2024},
      month = {jul},
      publisher = {IOP Publishing},
      volume = {26},
      number = {7},
      pages = {073017},
      author = {Galetsky, Vladlen and Julià Farré, Pol and Ghosh, Soham and Deppe, Christian and Ferrara, Roberto}
    }
