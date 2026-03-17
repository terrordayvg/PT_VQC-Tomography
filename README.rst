=========================================================================================================================
``Variational Process Tomography`` - A repository to perform quantum attacks with novel variational algorithms (PT_VQC and U-VQSVD)
=========================================================================================================================


.. image:: https://dl.circleci.com/status-badge/img/circleci/5ZWV663xqw4uDT8KDmJgpW/G4piVvQ66XDUHGX4Az1BJj/tree/circleci-project-setup.svg?style=shield&circle-token=41de148cb83684dd3c53509e74c3048071434118
        :target: https://dl.circleci.com/status-badge/redirect/circleci/5ZWV663xqw4uDT8KDmJgpW/G4piVvQ66XDUHGX4Az1BJj/tree/circleci-project-setup



.. image:: https://codecov.io/gh/terrordayvg/PT_VQC-Tomography/graph/badge.svg?token=880RTY0T96
        :target: https://codecov.io/gh/terrordayvg/PT_VQC-Tomography

.. image:: https://img.shields.io/badge/python-3.11-blue.svg
        :target: https://www.python.org/downloads/release/python-3110/

Docker - Installation of environment to run the code
-----

        1) Install Docker
        
        2) Build docker in cmd with command eg.: docker build -t qst-project .

        3) Run docker with pytest for testing functions with command eg.: docker run qst-project -m pytest Testing

        4) Run docker for chosen file eg. process_tomography.py file, the command should be: docker run qst-project Process_tomography/process_tomography_code.py


(Alternative - if not using Docker) Installation of required libraries

::

    install -r requirements.txt


Content
-----


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

        matplotlib==3.10.8
        numpy==2.4.3
        qiskit==1.1.2
        qiskit-aer==0.17.2
        scipy==1.17.1
        sympy==1.14.0
        pytest==9.0.2
        pluggy==1.6.0 
        tensorflow==2.21.0



How to Cite?
===========

This work generates the PT_VQC and U-VQSVD algorithms, if you use this work, please cite the following paper:

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
