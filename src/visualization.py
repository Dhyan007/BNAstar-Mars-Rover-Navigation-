import matplotlib.pyplot as plt
import networkx as nx


def get_neural_network_positions(layers):

    positions = {}

    for layer_index, layer in enumerate(layers):

        number_of_nodes = len(layer)

        for node_position, node in enumerate(layer):

            x = layer_index * 3

            if number_of_nodes == 1:

                y = 0

            else:

                y = (
                    node_position
                    - (number_of_nodes - 1) / 2
                ) * 1.5

            positions[node] = (x, y)

    return positions


def visualize_search_space(G, layers):

    positions = get_neural_network_positions(
        layers
    )

    plt.figure(
        figsize=(18, 10)
    )

    nx.draw_networkx_edges(
        G,
        positions,
        arrows=False,
        width=0.8,
        alpha=0.5
    )

    nx.draw_networkx_nodes(
        G,
        positions,
        node_size=700,
        node_color="white",
        edgecolors="black",
        linewidths=1.5
    )

    labels = {}

    for node in G.nodes:

        image_index = G.nodes[node].get("image_index", "N/A")
        astar_cost = G.nodes[node].get("astar_cost", None)

        if astar_cost is not None:
            labels[node] = (
                f"{node}\n"
                f"Img: {image_index}\n"
                f"Cost: {astar_cost:.3f}"
            )
        else:
            labels[node] = (
                f"{node}\n"
                f"Img: {image_index}"
            )

    nx.draw_networkx_labels(
        G,
        positions,
        labels,
        font_size=7
    )
    plt.title(
        "Mars Rover A* Search Space"
    )

    plt.axis("off")
    plt.tight_layout()
    plt.show()


def visualize_rover_path(
    G,
    layers,
    path
):

    positions = get_neural_network_positions(
        layers
    )

    plt.figure(
        figsize=(20, 12)
    )

    selected_edges = []

    for i in range(len(path) - 1):

        edge = (
            path[i],
            path[i + 1]
        )

        selected_edges.append(
            edge
        )

    nx.draw_networkx_edges(
        G,
        positions,
        arrows=False,
        width=0.8,
        alpha=0.25
    )

    nx.draw_networkx_edges(
        G,
        positions,
        edgelist=selected_edges,
        arrows=False,
        width=4,
        alpha=1.0
    )

    nx.draw_networkx_nodes(
        G,
        positions,
        node_size=750,
        node_color="white",
        edgecolors="black",
        linewidths=1.5
    )

    nx.draw_networkx_nodes(
        G,
        positions,
        nodelist=[path[0]],
        node_size=900,
        node_color="lightgreen",
        edgecolors="black",
        linewidths=2
    )

    nx.draw_networkx_nodes(
        G,
        positions,
        nodelist=[path[-1]],
        node_size=900,
        node_color="lightcoral",
        edgecolors="black",
        linewidths=2
    )

    node_labels = {
        node: str(node)
        for node in G.nodes
    }

    nx.draw_networkx_labels(
        G,
        positions,
        labels=node_labels,
        font_size=8
    )

    edge_labels = {}

    for u, v, data in G.edges(
        data=True
    ):

        if (
            "image_index" in data
            and "astar_cost" in data
        ):

            edge_labels[(u, v)] = (
                f"Img: {data['image_index']}\n"
                f"Cost: {data['astar_cost']:.3f}"
            )

    nx.draw_networkx_edge_labels(
        G,
        positions,
        edge_labels=edge_labels,
        font_size=7,
        label_pos=0.5
    )

    plt.title(
        "Mars Rover A* Search and Selected Path",
        fontsize=16
    )

    plt.axis("off")

    plt.tight_layout()

    plt.show()
