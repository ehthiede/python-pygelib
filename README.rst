========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - |
        |
    * - package
      - | |commits-since|
.. |docs| image:: https://readthedocs.org/projects/python-pygelib/badge/?style=flat
    :target: https://readthedocs.org/projects/python-pygelib
    :alt: Documentation Status

.. |commits-since| image:: https://img.shields.io/github/commits-since/ehthiede/python-pygelib/v0.0.0.svg
    :alt: Commits since latest release
    :target: https://github.com/ehthiede/python-pygelib/compare/v0.0.0...master



.. end-badges

Python interface to GElib

* Free software: MIT license

Installation
============

::

    pip install pygelib

You can also install the in-development version with::

    pip install https://github.com/ehthiede/python-pygelib/archive/master.zip


Documentation
=============


https://python-pygelib.readthedocs.io/


Development
===========

To run all the tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
