import networkx as nx


def create_search_space(layer_sizes):

    G = nx.DiGraph()

    node_id = 0
    layers = []

    for layer_size in layer_sizes:

        current_layer = []

        for _ in range(layer_size):

            G.add_node(node_id)

            current_layer.append(node_id)

            node_id += 1

        layers.append(current_layer)

    for i in range(len(layers) - 1):

        current_layer = layers[i]
        next_layer = layers[i + 1]

        current_size = len(current_layer)
        next_size = len(next_layer)

        for j, source in enumerate(current_layer):

            center = round(
                j * (next_size - 1)
                / max(current_size - 1, 1)
            )

            target_1 = next_layer[
                min(center, next_size - 1)
            ]

            G.add_edge(
                source,
                target_1
            )

            if next_size > 1:

                if center < next_size - 1:
                    target_2 = next_layer[center + 1]

                else:
                    target_2 = next_layer[center - 1]

                if target_2 != target_1:
                    G.add_edge(
                        source,
                        target_2
                    )

    return G, layers
