.. Main documentation page file, determines the overall layout of the static documentation .html page (after it is
   rendered with Sphinx) and allows linking additional sub-pages. Use it to build the skeleton of the documentation
   website.

.. Includes the Welcome page. This is the page that will be displayed whenever the user navigates to the documentation
   website.
.. include:: welcome.rst

.. Adds the left-hand-side navigation panel to the documentation website. Uses the API file to generate content list.
.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :hidden:

   api

Index
==================
* :ref:`genindex`
