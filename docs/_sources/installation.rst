Installation and model access
=============================

Requirements
------------

Use Python 3.10 or newer. A CUDA GPU is recommended for SAM 3 inference; CPU
inference is supported but can be slow. Projection and reuse of saved labels
need only the base package.

Install from PyPI:

.. code-block:: bash

   python -m pip install "segment-lidar[sam3]"

For the current source checkout:

.. code-block:: bash

   git clone https://github.com/Yarroudh/segment-lidar.git
   cd segment-lidar
   python -m pip install -e ".[sam3]"

Install ``segment-lidar[csf]`` as well if you need cloth simulation ground
filtering. For a CUDA installation of PyTorch, follow the `official PyTorch
selector <https://pytorch.org/get-started/locally/>`_ for your system.

Verify the base package:

.. code-block:: bash

   python -c "from segment_lidar import SamLidar, TopView, PinholeView; print('ready')"

SAM 3 access
------------

The default ``facebook/sam3`` model is gated on Hugging Face. A token alone
does not grant access:

1. Sign in and request access on the `SAM 3 model page
   <https://huggingface.co/facebook/sam3>`_. Accept the model conditions.
2. Wait for approval if the model owner reviews the request.
3. Create a `Hugging Face token <https://huggingface.co/settings/tokens>`_ with
   read permission from the *same approved account*.
4. Set ``HF_TOKEN`` in the runtime environment or pass ``hf_token`` to
   :class:`~segment_lidar.samlidar.SamLidar`.

For a shell session, enter the token without echoing it:

.. code-block:: bash

   read -rs HF_TOKEN
   export HF_TOKEN

Model files are downloaded to the Hugging Face cache at the first inference
call. Keep the token out of scripts, notebooks, images, and commits. Using
``image_exists=True`` and ``label_exists=True`` to transfer saved labels does
not load SAM 3 or require a token.

Docker
------

The repository's Dockerfile installs the SAM 3 extra by default:

.. code-block:: bash

   docker build -t segment-lidar:local .

Create a local ``.env`` file containing ``HF_TOKEN=your_token_here`` and run a
script from a mounted ``data`` directory:

.. code-block:: bash

   docker run --rm --gpus all --env-file .env \
     -v "$PWD/data:/data" segment-lidar:local python /data/process.py

The ``--gpus all`` option requires the NVIDIA container runtime. Omit it for
CPU inference. Build with ``--build-arg INSTALL_EXTRAS=`` for projection and
saved-label workflows that do not need SAM 3.

Building this documentation
---------------------------

Install the documentation extra, then build the GitHub Pages site from its
maintained source:

.. code-block:: bash

   python -m pip install -e ".[docs]"
   python -m sphinx -b html -W --keep-going docs/_sources docs

The ``docs`` directory is the published output. The editable source is in
``docs/_sources``.
