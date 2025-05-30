from docutils.parsers.rst.directives.images import Figure

class SupplementalFigure(Figure):
    def run(self):
        return super().run()

def setup(app):
    app.add_directive("suppfigure", SupplementalFigure)
    app.connect("builder-inited", lambda app: app.config.numfig_format.update({
        'suppfigure': 'Figure S%s'
    }))
    return {"version": "1.0", "parallel_read_safe": True}

