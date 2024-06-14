{{entry.qualified_name}}
{{'=' * (entry.qualified_name | length)}}

.. data:: {{entry.qualified_name}}
   :annotation:

   Instance of :class:`{{entry.parent_type}}`

   .. rubric:: Registered types

   .. list-table::
      :widths: 25 75

{% for type in entry.registered_types %}
      * - ``{{type.keyword}}``
        - :class:`{{type.cls}}`
{% endfor %}
