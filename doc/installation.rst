Installation
============

This section describes how to install ``CubeAI`` on your machine.

Dependencies
-------------

``CubeAI`` has several dependecies that should be available before installing the library. These are

- CMake
- Python >= 3.8
- `boost C++ libraries <https://www.boost.org/>`_
- `PyTorch C++ bindings <https://pytorch.org/>`_
- `Blaze <https://bitbucket.org/blaze-lib/blaze/src/master/>`_ (version >= 3.8) check `here <https://bitbucket.org/blaze-lib/blaze/wiki/Configuration%20and%20Installation>`_ how to configure and install Blaze
- Blas library, e.g. OpenBLAS (required by Blaze)
- `rlenvs_from_cpp <https://github.com/pockerman/rlenvs_from_cpp>`_
- `nlohmann_json <https://github.com/nlohmann/json>`_


Note that ``CubeAI`` requires PyTorch binaries with C++ 11 support. 
Moreover, if you choose PyTorch with CUDA then ```cuDNN``` library is also required. 
This is a runtime library containing primitives for deep neural networks.

Furthermore, ``CubeAI`` has the following integrated dependencies

- `matplotlib-cpp <https://github.com/lava/matplotlib-cpp>`_
- `better-enums <https://github.com/aantron/better-enums>`_

Assuming that all dependencies are properly installed, you can use the common cmake process
to install the library. Namely,

.. code-block:: console

	 mkdir build && cd build
	 cmake ..
	 make install
	 
Configuration
-------------

Not all functionality may be relevant to what you. In order to ensure small binaries, you can
configure the library according to what you need. Currently, this is done in primitive manner; you
will have to edit the ``CMakeLists.txt`` file yourself and activate/deactivate functionality
	 
If you are using ```rl_envs_from_cpp``` you need to export the path to the Python version you are using. For ecample:

.. code-block:: console
	
	export CPLUS_INCLUDE_PATH="$CPLUS_INCLUDE_PATH:/usr/include/python3.8/"
	
or 

.. code-block:: console

	export CPLUS_INCLUDE_PATH="$CPLUS_INCLUDE_PATH:/usr/include/python3.10/"


Depending on the values of the ```CMAKE_BUILD_TYPE```, the produced shared library will 
be installed in ```CMAKE_INSTALL_PREFIX/dbg/``` or ```CMAKE_INSTALL_PREFIX/opt/``` directories.

Issues with C++ installation
----------------------------

- Could not find boost. On a Ubuntu machine you can install the boost libraries as follows

.. code-block:: console
	
	sudo apt-get install libboost-all-dev


- ``pyconfig.h`` not found

In this case we may have to export the path to your Python library directory as shown above.

- Problems with Blaze includes

``cubeai`` is using Blaze-3.8. As of this version the ``FIND_PACKAGE( blaze )`` command does not populate ``BLAZE_INCLUDE_DIRS``  therefore you manually have to set the variable appropriately for your system. So edit the project's ``CMakeLists.txt`` file and populate appropriately the variable ``BLAZE_INCLUDE_DIRS``.

- Could not find BLAS 

The ```Blaze``` library depends on BLAS so it has to be installed. 
On a Ubuntu machine this can be done as follows

.. code-block:: console

	sudo apt-get install libblas-dev liblapack-dev

Depending on how you configure things, you may need to install OpenBLAS

.. code-block:: console
	
	sudo apt-get install libopenblas-dev

- No CMAKE_CUDA_COMPILER could be found

In this case CMake could not found the Cuda compiler. On a Ubuntu machine,
you can check where ``nvcc`` is installed using ``which nvcc``. You can
install the Cuda compiler by installing the Cuda toolkit. 

.. code-block:: console

	sudo apt install nvidia-cuda-toolkit

And then set the variable ``CMAKE_CUDA_COMPILER`` to the directory with the ``nvcc`` executable.
For a Ubuntu machine this may be ``/usr/local/cuda/bin`` 


Generate documentation
----------------------

You will need `Sphinx <https://www.sphinx-doc.org/en/master/>`_ in order to generate the API documentation. Assuming that Sphinx is already installed
on your machine execute the following commands (see also `Sphinx tutorial <https://www.sphinx-doc.org/en/master/tutorial/index.html>`_). 

.. code-block:: console

	sphinx-quickstart docs
	sphinx-build -b html docs/source/ docs/build/html




