Installation
============

Requirements
------------

* Python 3.8 or higher
* PyTorch 2.0 or higher

Install from PyPI
-----------------

.. code-block:: bash

    pip install pytorch-recsys-framework

Development Installation
------------------------

1. Clone the repository:

.. code-block:: bash

    git clone https://github.com/louiswang524/pytorch-recsys.git
    cd pytorch-recsys

2. Create a virtual environment:

.. code-block:: bash

    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install in development mode:

.. code-block:: bash

    pip install -e .[dev]

4. Install pre-commit hooks:

.. code-block:: bash

    pre-commit install

Verify Installation
-------------------

.. code-block:: python

    import pytorch_recsys
    print(pytorch_recsys.__version__)