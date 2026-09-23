import numpy as np
import os
import random
import pandas as pd
from IPython.display import display

from .segmentation import get_random_images_and_probabilities
from .sensors import add_sensor_data
from .jev import add_jev_results
from .bayesian_network import add_bn_results


def add_astar_cost(candidate_df):

    candidate_df = candidate_df.copy()

    success = np.clip(
        candidate_df["bn_traversal_success"],
        1e-6,
        1.0
    )

    candidate_df["astar_cost"] = -np.log(success)

    return candidate_df


def evaluate_two_candidates(images_path, model):

    candidate_results = (
        get_random_images_and_probabilities(
            images_path,
            model,
            num_images=2
        )
    )

    candidate_df = pd.DataFrame(
        candidate_results
    )

    candidate_df = add_sensor_data(
        candidate_df
    )

    candidate_df = add_jev_results(
        candidate_df
    )

    candidate_df = add_bn_results(
        candidate_df
    )

    candidate_df = add_astar_cost(
        candidate_df
    )

    return candidate_df


def run_rover_simulation(
    G,
    layers,
    images_path,
    model
):

    current_node = layers[0][0]
    goal_node = layers[-1][0]

    path = [current_node]

    movement_history = []

    total_cost = 0.0

    print("=" * 70)
    print("MARS ROVER A* SIMULATION")
    print("=" * 70)

    print(f"Start node : {current_node}")
    print(f"Goal node  : {goal_node}")

    while current_node != goal_node:

        print("\n")
        print("=" * 70)
        print(f"CURRENT NODE: {current_node}")
        print("=" * 70)

        next_nodes = list(
            G.successors(current_node)
        )

        if len(next_nodes) == 0:
            raise RuntimeError(
                f"Node {current_node} has no outgoing paths."
            )

        print(
            f"Possible next nodes: {next_nodes}"
        )

        num_candidates = min(
            2,
            len(next_nodes)
        )

        candidate_results = (
            get_random_images_and_probabilities(
                images_path,
                model,
                num_images=num_candidates
            )
        )

        candidate_df = pd.DataFrame(
            candidate_results
        )

        candidate_df = add_sensor_data(
            candidate_df
        )

        candidate_df = add_jev_results(
            candidate_df
        )

        candidate_df = add_bn_results(
            candidate_df
        )

        candidate_df = add_astar_cost(
            candidate_df
        )

        for i, next_node in enumerate(next_nodes):

            if i >= len(candidate_df):
                break

            candidate = candidate_df.iloc[i]

            G.edges[
                current_node,
                next_node
            ]["image_index"] = int(
                candidate["image_index"]
            )

            G.edges[
                current_node,
                next_node
            ]["image"] = candidate["image"]

            G.edges[
                current_node,
                next_node
            ]["bn_traversal_success"] = float(
                candidate["bn_traversal_success"]
            )

            G.edges[
                current_node,
                next_node
            ]["astar_cost"] = float(
                candidate["astar_cost"]
            )

        display_columns = [
            "image_index",
            "image",
            "bn_traversal_success",
            "astar_cost"
        ]

        print("\nCandidate evaluation:")

        display(
            candidate_df[display_columns]
        )

        best_index = candidate_df[
            "astar_cost"
        ].idxmin()

        best_candidate = candidate_df.loc[
            best_index
        ]

        selected_position = (
            candidate_df.index.get_loc(
                best_index
            )
        )

        selected_node = next_nodes[
            selected_position
        ]

        selected_cost = float(
            best_candidate["astar_cost"]
        )

        selected_success = float(
            best_candidate[
                "bn_traversal_success"
            ]
        )

        selected_image_index = int(
            best_candidate["image_index"]
        )

        total_cost += selected_cost

        movement = {
            "from_node": current_node,
            "to_node": selected_node,
            "image_index": selected_image_index,
            "image": best_candidate["image"],
            "bn_traversal_success": selected_success,
            "astar_cost": selected_cost,
            "total_cost": total_cost
        }

        movement_history.append(
            movement
        )

        G.edges[
            current_node,
            selected_node
        ]["selected"] = True

        print("\nSELECTED PATH")
        print("-" * 40)

        print(
            f"From node       : {current_node}"
        )

        print(
            f"To node         : {selected_node}"
        )

        print(
            f"Image index     : {selected_image_index}"
        )

        print(
            f"BN success      : "
            f"{selected_success:.4f}"
        )

        print(
            f"A* cost         : "
            f"{selected_cost:.4f}"
        )

        print(
            f"Accumulated cost: "
            f"{total_cost:.4f}"
        )

        current_node = selected_node

        path.append(
            current_node
        )

    print("\n")
    print("=" * 70)
    print("DESTINATION REACHED")
    print("=" * 70)

    print(
        "Path:",
        " → ".join(
            map(str, path)
        )
    )

    print(
        f"Total cost: {total_cost:.4f}"
    )

    return path, movement_history
