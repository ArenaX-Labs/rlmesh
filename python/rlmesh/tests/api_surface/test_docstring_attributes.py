import griffe
from rlmesh_api_surface.api_surface import _render_docstring


def test_attribute_descriptions_render_as_fields():
    documentation = griffe.Docstring(
        """Action fields.

        Attributes:
            components: Actuators in vector order.
            clip: Optional final bounds.
        """
    )
    rendered = _render_docstring(documentation)
    assert "**Attributes**" in rendered
    assert "`components`: Actuators in vector order." in rendered
    assert "`clip`: Optional final bounds." in rendered
    assert "object at 0x" not in rendered


def test_admonition_renders_description():
    rendered = _render_docstring(
        griffe.Docstring("Convert a tensor.\n\nNote:\n    The result shares memory.")
    )
    assert "The result shares memory." in rendered
    assert "object at 0x" not in rendered
