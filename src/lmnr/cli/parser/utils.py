# Convert a list of handles to a map of input handle names
# to their respective values
import uuid
from .nodes import Handle


def map_handles(
    inputs: list[Handle],
    inputs_mappings: dict[uuid.UUID, uuid.UUID],
    output_handle_id_to_node_name: dict[str, str],
) -> list[tuple[str, str]]:
    mapping = []

    for to, from_ in inputs_mappings.items():
        for input in inputs:
            if input.id == to:
                mapping.append((input.name, from_))
                break
        else:
            raise ValueError(f"Input handle {to} not found in inputs")

    return [
        (input_name, output_handle_id_to_node_name[str(output_id)])
        for input_name, output_id in mapping
    ]


def replace_spaces_with_underscores(s: str):
    spaces = [
        "\u0020",
        "\u00A0",
        "\u1680",
        "\u2000",
        "\u2001",
        "\u2002",
        "\u2003",
        "\u2004",
        "\u2005",
        "\u2006",
        "\u2007",
        "\u2008",
        "\u2009",
        "\u200A",
        "\u200B",
        "\u202F",
        "\u205F",
        "\u3000",
    ]
    return s.translate({ord(space): "_" for space in spaces})
