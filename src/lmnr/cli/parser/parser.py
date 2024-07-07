from .nodes.types import node_from_dict


def runnable_graph_to_template_vars(graph: dict) -> dict:
    """
    Convert a runnable graph to template vars to be rendered in a cookiecutter context.
    """
    node_id_to_node_name = {}
    output_handle_id_to_node_name: dict[str, str] = {}
    for node in graph["nodes"].values():
        node_id_to_node_name[node["id"]] = node["name"]
        for handle in node["outputs"]:
            output_handle_id_to_node_name[handle["id"]] = node["name"]

    tasks = []
    for node_obj in graph["nodes"].values():
        node = node_from_dict(node_obj)
        handles_mapping = node.handles_mapping(output_handle_id_to_node_name)
        node_type = node.node_type()
        tasks.append(
            {
                "name": node.name,
                "function_name": f"run_{node.name}",
                "node_type": node_type,
                "handles_mapping": handles_mapping,
                # since we map from to to from, all 'to's won't repeat
                "input_handle_names": [
                    handle_name for (handle_name, _) in handles_mapping
                ],
                "handle_args": ", ".join(
                    [
                        f"{handle_name}: NodeInput"
                        for (handle_name, _) in handles_mapping
                    ]
                ),
                "prev": [],
                "next": [],
                "config": node.config(),
            }
        )

    for to, from_ in graph["pred"].items():
        # TODO: Make "tasks" a hashmap from node id (as str!) to task
        to_task = [task for task in tasks if task["name"] == node_id_to_node_name[to]][
            0
        ]
        from_tasks = []
        for f in from_:
            from_tasks.append(
                [task for task in tasks if task["name"] == node_id_to_node_name[f]][0]
            )

        for from_task in from_tasks:
            to_task["prev"].append(from_task["name"])
            from_task["next"].append(node_id_to_node_name[to])

    # Return as a hashmap due to cookiecutter limitations, investigate later.
    return {task["name"]: task for task in tasks}
