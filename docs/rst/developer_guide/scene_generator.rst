.. _sec-developer_guides-scene_generator:

Scene generator design
======================

Eradiate's scene generator design is tightly coupled with its radiometric kernel
interface. This document presents the general underlying design and briefly
explains how to create new scene elements.

.. warning::
   The design presented in this page has several known issues which we hope to
   overcome in the future.

Fundamentals
------------

Eradiate splits scene definition into
:ref:`scene elements <sec-user_guide-basic_concepts-scene_elements>`.
Scene elements inherit the :class:`.SceneElement` interface and are assembled
into a tree which is traversed to assemble a :term:`kernel dictionary template`
and a :term:`parameter update map template` during the initialization phase of
the processing pipeline. We will elaborate on the traversal step later.

The :class:`.SceneElement` interface associates each tree node with a parameter
update map template contribution through its :attr:`.SceneElement.params`
property. Scene elements are categorized based on how they are expanded during
traversal:

Node scene element [:class:`.NodeSceneElement`]
    Node scene elements expand as proper nodes in the Mitsuba scene graph and
    can be initialized using a kernel dictionary entry. They provide
    abstractions very close to Mitsuba's. Typical examples are BSDFs
    [:mod:`eradiate.scenes.bsdfs`] and spectra [:mod:`eradiate.scenes.spectra`].
    Node scene elements provide a scene dictionary template contribution with
    their :attr:`.NodeSceneElement.template` property. Optionally, child objects
    can also be declared with the :attr:`.NodeSceneElement.objects` property
    for an easy forwarding of traversal.

Instance scene element [:class:`.InstanceSceneElement`]
    Instance scene elements expand as a single Mitsuba object instance. They
    cannot be expressed as a kernel scene dictionary part. They return the
    generated Mitsuba object through their
    :attr:`.InstanceSceneElement.instance` property.

Composite scene element [:class:`.CompositeSceneElement`]
    Composite scene elements expand to multiple Mitsuba scene tree nodes and can
    be expressed as kernel scene dictionary parts. The kernel dictionary
    template contribution is provided by the
    :attr:`.CompositeSceneElement.template`. Child objects are declared using
    the :attr:`.CompositeSceneElement.objects` property.

The traversal protocol is different for each of these three elements types,
which all provide a different implementation of the
:meth:`.SceneElement.traverse` method.

All scene elements can be recursively traversed using the :func:`.traverse`
function. This function outputs a pair consisting of a
:term:`kernel dictionary template` and a :term:`parameter update map template`,
which are then used by the :class:`.Experiment` to assemble a Mitsuba scene and
update it as part of the parametric loop.

Writing a new scene element class
---------------------------------

1. Decide whether the scene element is a :class:`.NodeSceneElement`,
   an :class:`.InstanceSceneElement` or a :class:`.CompositeSceneElement`.
   In most cases, the choice is constrained by the scene element subtype.
   For example, :class:`.BSDF`\ s are all :class:`.NodeSceneElement`\ s,
   :class:`.Surface`\ s are all :class:`.CompositeSceneElement`\ s, etc.
2. Derive a new class from the selected type and make sure that the properties
   mentioned in the `Fundamentals`_ section are all implemented.

   .. list-table::
      :widths: 1 2
      :header-rows: 1

      * - Type
        - Properties
      * - :class:`.NodeSceneElement`
        - :attr:`~.NodeSceneElement.template`,
          :attr:`~.NodeSceneElement.params` [,
          :attr:`~.NodeSceneElement.objects`]
      * - :class:`.InstanceSceneElement`
        - :attr:`~.InstanceSceneElement.instance`,
          :attr:`~.InstanceSceneElement.params`
      * - :class:`.CompositeSceneElement`
        - :attr:`~.CompositeSceneElement.template`,
          :attr:`~.CompositeSceneElement.params` [,
          :attr:`~.CompositeSceneElement.objects`]

3. When writing unit tests, make sure to include a basic sanity check using the
   :func:`.check_scene_element` function.
