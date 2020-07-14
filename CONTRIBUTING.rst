************
Contributing
************

We greatly appreciate your contributions!

The following is a collection of guidelines for contributing to the code repository for the
O'Reilly publication `"Building Machine Learning Pipelines" <https://www.buildingmlpipelines.com/>`_
by Hannes Hapke & Catherine Nelson.

If you found an error in the book, please report it at
https://www.oreilly.com/catalog/errata.csp?isbn=0636920260912.

Types of contributions:
=======================

.. _issue:

1. Reporting bugs
-----------------

Please report bugs at
https://github.com/Building-ML-Pipelines/building-machine-learning-pipelines/issues.

If you are reporting a bug, we ask you to include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

2. Fixing bugs
--------------

Bug fixes to our code are always welcome. All bug fixes should be connected to an issue.
If you plan to fix a bug, please use the appropriate issue to briefly tell us what you have in mind.
If you have found a bug and there is no issue yet, please start by filing an issue_.

You can also look through the `GitHub issues <https://github.com/Building-ML-Pipelines/building-machine-learning-pipelines/issues>`_
for bugs. Anything tagged with "bug" is open to whoever wants to fix it.

See the section `Contributing to this repository`_ below for instructions on how to
set up your environment and create a pull request.

Providing general feedback
--------------------------

The best way to send general feedback is to file an issue_.
You can submit errata for the publication here: https://www.oreilly.com/catalog/errata.csp?isbn=0636920260912

Contributing to this repository
===============================

Setting up your development environment
---------------------------------------

This is how you set up a local development environment to work on the code:

Create a fork of the
`Github repository <https://github.com/Building-ML-Pipelines/building-machine-learning-pipelines>`_.

Clone your fork locally::

   $ git clone git@github.com:[YOUR USERNAME]/building-machine-learning-pipelines.git

Create a branch for your contribution::

   $ git checkout -b name-of-your-contribution

Create a virtualenv to separate your Python dependencies::

   $ virtualenv .env && source .env/bin/activate

Download and install all dependencies required for development::

   $ make develop

This will automatically download and configure all required dependencies.
Your local environment is now ready  for you to begin making your changes.

Testing and linting
-------------------

When you're done making changes, please make sure that your code passes style and unit tests.

You can run linting and all testing with this command::

    $ make test

In case you want to run only specific tests instead of all available tests, you can call
Pytest directly and combine it with a substring. Pytest will only run tests with names
matching the substring::

    $ pytest -k <substring> -v

You are welcome to add your name to AUTHORS.rst before committing your code!

Submitting a pull request
-------------------------

Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Description of your changes."
    $ git push origin name-of-your-contribution

Check that your pull request meets these guidelines before you submit it:

1. If the pull request adds or changes functionality, it should include updated tests.
2. The pull request should work with Python 3.6, 3.7 and 3.8. Make sure that
   all tests run by ``make test`` pass.
