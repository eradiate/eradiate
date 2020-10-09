from docutils import nodes
from docutils.parsers.rst import directives
from docutils.parsers.rst.directives.tables import Table
from docutils.statemachine import ViewList
from sphinx.util import nested_parse_with_titles
from tinydb import TinyDB, Query
from tinydb.storages import MemoryStorage

from eradiate.scenes.core import SceneElementFactory

factory_db = TinyDB(storage=MemoryStorage)
factory_db.insert_multiple([
    {
        "key": key,
        "module": str(value.__module__),
        "cls_name": str(value.__name__)
    }
    for key, value in SceneElementFactory.registry.items()
])


class FactoryTable(Table):
    # https://github.com/sphinx-contrib/documentedlist/blob/master/sphinxcontrib/documentedlist.py
    option_spec = {
        "modules": directives.unchanged,
        "sections": directives.flag
    }

    def __init__(self, *args, **kwargs):
        super(FactoryTable, self).__init__(*args, **kwargs)
        self.headers = None
        self.max_cols = None
        self.col_widths = None

    def run(self):
        if self.content:
            error = self.state_machine.reporter.error(
                "The factorytable directive does not know what to do with "
                "provided content",
                nodes.literal_block(self.block_text, self.block_text),
                line=self.lineno
            )
            return [error]

        # Process list of modules for filtering
        modules = self.options.get("modules", None)
        if modules is None:
            modules = SceneElementFactory._submodules
        else:
            modules = [x.strip() for x in  modules.split(",")]

        if not set(modules) <= set(SceneElementFactory._submodules):
            error = self.state_machine.reporter.error(
                f"The following requested modules are not inspected by the " 
                f"Eradiate factory: "
                f"{', '.join(set(modules) - set(SceneElementFactory._submodules))}",
                nodes.literal_block(self.block_text, self.block_text),
                line=self.lineno
            )
            return [error]

        # Get the list containing the documentation
        factory_submodules = modules
        show_sections = "sections" in self.options

        member = []

        for submodule in factory_submodules:
            if show_sections:
                member.append([f":bolditalic:`In submodule` "
                               f":mod:`~eradiate.scenes.{submodule}`",
                               "—"])

            for element_info in factory_db.search(Query().module.matches(
                    rf"^eradiate\.scenes\.{submodule}.*")):
                factory_keyword = f"``{element_info['key']}``"
                class_module = element_info["module"]
                class_name = element_info["cls_name"]
                class_reference = f":class:`~{class_module}.{class_name}`"
                member.append([factory_keyword, class_reference])

        # Set headers
        self.headers = ["Factory keyword", "Class"]
        self.max_cols = len(self.headers)

        # This works around an apparently poorly documented change in docutils
        # v13.1. I _think_ this should work with both versions, but please file
        # a bug on github if it doesn't.
        rval = self.get_column_widths(self.max_cols)
        if len(rval) == 2 and isinstance(rval[1], list):
            self.col_widths = rval[1]
        else:
            self.col_widths = rval

        table_body = member
        title, messages = self.make_title()
        table_node = self.build_table(table_body)
        self.add_name(table_node)
        if title:
            table_node.insert(0, title)
        return [table_node] + messages

    def get_rows(self, table_data):
        rows = []
        groups = []
        for row in table_data:
            trow = nodes.row()
            for idx, cell in enumerate(row):
                entry = nodes.entry()

                # Process rst
                # See https://stackoverflow.com/questions/34350844/how-to-add-rst-format-in-nodes-for-directive
                rst = ViewList()
                rst.append(str(cell), "", 0)
                parsed_node = nodes.section()
                parsed_node.document = self.state.document
                nested_parse_with_titles(self.state, rst, parsed_node)

                entry += parsed_node[0]
                trow += entry

            rows.append(trow)

        return rows, groups

    def build_table(self, table_data):
        table = nodes.table()
        tgroup = nodes.tgroup(cols=len(self.headers))
        table += tgroup

        tgroup.extend(
            nodes.colspec(colwidth=col_width, colname='c' + str(idx))
            for idx, col_width in enumerate(self.col_widths)
        )

        thead = nodes.thead()
        tgroup += thead

        row_node = nodes.row()
        thead += row_node
        row_node.extend(nodes.entry(h, nodes.paragraph(text=h))
                        for h in self.headers)

        tbody = nodes.tbody()
        tgroup += tbody

        rows, groups = self.get_rows(table_data)
        tbody.extend(rows)
        table.extend(groups)

        return table


def factory_key(name, rawtext, text, lineno, inliner, options={}, content=[]):
    """Link factory key to corresponding class.

    https://doughellmann.com/blog/2010/05/09/defining-custom-roles-in-sphinx/

    :param name: The role name used in the document.
    :param rawtext: The entire markup snippet, with role.
    :param text: The text marked with the role.
    :param lineno: The line number where rawtext appears in the input.
    :param inliner: The inliner instance that called us.
    :param options: Directive options for customization.
    :param content: The directive content for customization.
    """
    key = text
    db_entries = factory_db.search(Query().key == key)
    if len(db_entries) == 0:
        msg = inliner.reporter.error(
            f"could not find factory key {key}",
            line=lineno
        )
        prb = inliner.problematic(rawtext, rawtext, msg)
        return [prb], [msg]

    db_entry = db_entries[0]
    mod, cls_name = db_entry["module"], db_entry["cls_name"]

    # Create link node
    # TODO: format as monospaced
    # TODO: not robust, retrieve class URI and use it
    # node = nodes.Text(text)
    node = nodes.reference(rawsource=rawtext, text=key,
                           refuri=f"{mod}.{cls_name}.html")

    return [node], []


def setup(app):
    app.add_directive("factorytable", FactoryTable)
    app.add_role("factorykey", factory_key)

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
