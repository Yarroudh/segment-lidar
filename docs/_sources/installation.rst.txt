Installation
============

This guide describes the steps to install **segment-lidar** using `PyPI <https://pypi.org/project/segment-lidar/>`__ or from source.

Step 1: Create an environment
-----------------------------

Before installing **segment-lidar**, you need to create an environment by
running the following commands:

.. code:: bash

   conda create -n samlidar python=3.9
   conda activate samlidar

This command will create a new Conda environment named **samlidar**. We recommend using **Python 3.9**, but feel free to test with other versions.

Please note that using a Conda environment is not mandatory, but it is highly recommended. Alternatively, you can use `virtualenv <https://virtualenv.pypa.io/en/latest/>`__.

Step 2: Install segment-lidar
-----------------------------

You can easily install **segment-lidar** from `PyPI <https://pypi.org/project/segment-lidar/>`__ using the following command:

.. code:: bash

    pip install segment-lidar

Or, you can install it from source:

.. code:: bash

    git clone https://github.com/Yarroudh/segment-lidar
    cd segment-lidar
    python setup.py install

To make sure that **segment-lidar** is installed correctly, you can run the following command:

.. code:: bash

    python -c "import segment_lidar; print(segment_lidar.__version__)"

If the installation is successful, you should see the version that you have installed.