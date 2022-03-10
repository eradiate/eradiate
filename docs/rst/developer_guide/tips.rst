.. _sec-developer_guide-maintainer_tips:

Maintainer tips
===============

Shallow submodule caveats
-------------------------

Eradiate uses Git submodules to ship some of its data. Over time, these can grow
and become large enough so that using a *shallow submodule*. Shallow clones
do not contain the entire history of the repository and are therefore more
lightweight, saving bandwidth upon cloning.

However, shallow clones can be difficult to work with, especially when one
starts branching. If a shallow submodule is missing a remote branch you'd expect
it to track,
`this post <https://stackoverflow.com/questions/23708231/git-shallow-clone-clone-depth-misses-remote-branches>`_
contains probably what you need to do:

.. code:: bash

   cd my-shallow-submodule
   git remote set-branches origin '*'
   git fetch -v
   git checkout the-branch-i-ve-been-looking-for

Tagging a commit for release
----------------------------

Eradiate picks up its version number using the `setuptools-scm <https://github.com/pypa/setuptools_scm>`_
package. Under the hood, it uses Git tags and the ``git describe`` command,
which only picks up annotated tags. To make sure that the tags will be
correctly picked up,
`make sure that they are annotated <https://stackoverflow.com/questions/4154485/git-describe-ignores-a-tag>`_
using

.. code:: bash

   git tag -a <your_tag_name>
