.. _sec-getting_started-docker:

Docker guide
============

In this section we discuss the usage of docker to speed up the development environment installation.

Docker is a containerization tool which help building and deploying applications on multiple operating systems.
It has the benefits of allowing the installation of multiple softwares with the same reproducible and simple procedure.

See the `installation guide <https://eradiate.readthedocs.io/en/latest/rst/getting_started/install.html>`_ to setup Eradiate without Docker

Motivations
-----------

Docker and other compatible containerization tools have become more and more popular in the recent years.

The Eradiate team wanted to provide an easy solution to allow fast installation of the project at least for test automation.

The Eradiate images may help users to bootstrap their Eradiate project by having a working environment within minutes on most modern operating systems.

The isolation provided by containers may be suitable to users who don't want to add the multiple Eradiate dependencies to their current working environment or to maintain an Eradiate development environment.

Docker can also be used to ship scripts importing/using the eradiate framework.

Installing Docker
-----------------

The Docker installation procedure is well documented and rather easy to follow for supported operating systems.

Please refer to the official documentation for `installing Docker <https://docs.docker.com/get-docker>`_ on your development machine.

After installing docker, you should be able to launch a simple docker example from a terminal:

.. code:: bash
    
    # Launch a simple hello world program from a container
    docker run hello-world

.. note::

    Depending on your installation, you may need to launch the Docker CLI with root access: 

    .. code:: bash

        # Launch a simple hello world program from a container
        sudo docker run hello-world

Please refer to the official `Docker Documentation <https://docs.docker.com/>`_ for more information on the CLI usage.

Eradiate Docker Images Structure
--------------------------------

Three different images are provided by Eradiate:
 - fxia/eradiate-kernel
 - fxia/eradiate
 - fxia/eradiate-jupyterlab

.. only:: latex

   .. figure:: ../../fig/docker-images-structure.png

.. only:: not latex

   .. figure:: ../../fig/docker-images-structure.svg

Eradiate Kernel
---------------

This image provides an environment with a ready to use Mitsuba 2 engine compiled for Eradiate

.. code:: bash
    
    # Launch bash on the eradiate-kernel container
    docker run -it fxia/eradiate-kernel bash

    # Run mitsuba inside the container
    mitsuba --help
    
This mitsuba2 engine has been compiled with the correct parameters for Eradiate.
It is used as a base for the other images, and is provided for testing.

Eradiate
--------

This image provides an installed version of Eradiate, with a modern python3 distribution

It can be used to run some tests or develop your application and creating your own Docker image.

.. code:: bash

    # Launch python on the eradiate container
    docker run -it fxia/eradiate python

.. code:: python

    # Import Eradiate and start playing
    import eradiate
    import eradiate.data as data

Building a custom Eradiate image
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Building an image can be a convenient way to ship an Eradiate project.

Create a file named Dockerfile. The following block illustrate some example Dockerfile content to build a container image based on eradiate.

The `myProjectScript.py` may import eradiate.

.. code::

    FROM fxia/eradiate

    RUN pip install myProjectDep1 myProjectDep2

    COPY ./myProjectScript.py /app/myProjectScript.py

    CMD python /app/myProjectScript.py

Please refer to the `official documentation <https://docs.docker.com/engine/reference/builder/>`_ for more details on how to write a Dockerfile.

Having this Dockerfile in the current directory, use the following commands to build the image, tag it `myproject`, and launch the container

.. code:: bash

    # Build the image
    docker build . --tag myproject

    # Run the built image
    docker run -it myproject

The image `myproject` may now be published on a public or private registry such as Docker Hub to be accessible to end users.

Please refer to `this documentation <https://docs.docker.com/docker-hub/publish/publish/>`_ to publish your images on Docker Hub.

Eradiate Jupyterlab
-------------------

The jupyterlab image can also be used for development, it exposes a server on which users can connect locally.

This server bundles everything needed to run Eradiate.

.. code:: bash

    # Launch an Eradiate ready Jupyterlab in a container
    docker run -p "8888:8888" --rm -it fxia/eradiate-jupyterlab

After downloading and launching the server, this command will various URLs for your web browser.
Here is an example output of the above command:

.. code::

    [W 2021-03-18 14:13:16.619 ServerApp] Unrecognized alias: 'allow_origin', it will have no effect.
    [I 2021-03-18 14:13:16.643 ServerApp] jupyterlab | extension was successfully linked.
    [I 2021-03-18 14:13:16.657 ServerApp] Writing notebook server cookie secret to /root/.local/share/jupyter/runtime/jupyter_cookie_secret
    [I 2021-03-18 14:13:16.833 ServerApp] nbclassic | extension was successfully linked.
    [I 2021-03-18 14:13:16.868 LabApp] JupyterLab extension loaded from /usr/local/lib/python3.8/dist-packages/jupyterlab
    [I 2021-03-18 14:13:16.868 LabApp] JupyterLab application directory is /usr/local/share/jupyter/lab        
    [I 2021-03-18 14:13:16.872 ServerApp] jupyterlab | extension was successfully loaded.
    [I 2021-03-18 14:13:16.876 ServerApp] nbclassic | extension was successfully loaded.
    [I 2021-03-18 14:13:16.877 ServerApp] Serving notebooks from local directory: /app
    [I 2021-03-18 14:13:16.877 ServerApp] Jupyter Server 1.4.1 is running at:
    [I 2021-03-18 14:13:16.877 ServerApp] http://611969a8b36a:8887/lab?token=4ec44260b1781a011ed75e0c9a47d18fe3bf0af5635f6732
    [I 2021-03-18 14:13:16.877 ServerApp]  or http://127.0.0.1:8887/lab?token=4ec44260b1781a011ed75e0c9a47d18fe3bf0af5635f6732
    [I 2021-03-18 14:13:16.877 ServerApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
    [C 2021-03-18 14:13:16.881 ServerApp]

        To access the server, open this file in a browser:
            file:///root/.local/share/jupyter/runtime/jpserver-7-open.html
        Or copy and paste one of these URLs:
            http://611969a8b36a:8887/lab?token=4ec44260b1781a011ed75e0c9a47d18fe3bf0af5635f6732
        or http://127.0.0.1:8887/lab?token=4ec44260b1781a011ed75e0c9a47d18fe3bf0af5635f6732

Please connect to the server via a web browser through the address with the 127.0.0.1 IP: `http://127.0.0.1:8887/lab?token=4ec44260b1781a011ed75e0c9a47d18fe3bf0af5635f6732`.
Other listed addresses are not exposed by the container and may fail to load properly.

.. note:: **Running multiple Jupyter instances**

    Users already operating one or multiple Jupyter instances on their machine may want to change the port bindings of the container: 

    .. code:: bash

        # Specify a port for the Eradiate Jupyterlab
        docker run -p "8887:8887" -e PORT=8887 --rm -it fxia/eradiate-jupyterlab
