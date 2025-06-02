from docutils import nodes
from docutils.parsers.rst.directives.images import Figure

class SupplementalFigure(Figure):
    def run(self):
        fig_nodes = super().run()
        for node in fig_nodes:
            if isinstance(node, nodes.figure):
                node['classes'].append('suppfigure')
                node['figtype'] = 'suppfigure'  # critical for numfig_format!
        return fig_nodes

def setup(app):
    app.add_directive("suppfigure", SupplementalFigure)

    def update_formats(app):
        app.config.numfig_format['suppfigure'] = 'Figure S%s'

    app.connect("builder-inited", update_formats)

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
