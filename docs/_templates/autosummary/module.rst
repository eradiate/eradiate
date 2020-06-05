{{ fullname | escape | underline }}

.. rubric:: Description

.. automodule:: {{ fullname }}

.. currentmodule:: {{ fullname }}

{% if classes %}
.. rubric:: Classes

.. autosummary::
    :toctree: .
    {% for class in classes %}
    {{ class }}
    {% endfor %}

{% endif %}

{% if functions %}
.. rubric:: Functions

.. autosummary::
    :toctree: .
    {% for function in functions %}
    {{ function }}
    {% endfor %}

{% endif %}

{% if exceptions %}
.. rubric:: Exceptions

.. autosummary::
    :toctree: .
    {% for exception in exceptions %}
    {{ exception }}
    {% endfor %}

{% endif %}

{% if attributes %}
.. rubric:: Attributes
    :toctree: .
    {% for attribute in attributes %}
    {{ attribute }}
    {% endfor %}

{% endif %}
