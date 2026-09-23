import numpy as np
import pandas as pd

SAFETY_STATES = [
    "Low",
    "Medium",
    "High"
]
BATTERY_STATES = [
    "Low",
    "Medium",
    "High"
]
TRAVERSAL_STATES = [
    "Failure",
    "Success"
]

SAFETY_CPT = {

    "Low": {
        "Low": 0.05,
        "Medium": 0.15,
        "High": 0.80
    },

    "Medium": {
        "Low": 0.15,
        "Medium": 0.35,
        "High": 0.50
    },

    "High": {
        "Low": 0.60,
        "Medium": 0.30,
        "High": 0.10
    }
}

BATTERY_CPT = {

    "Low": {
        "Low": 0.80,
        "Medium": 0.15,
        "High": 0.05
    },

    "Medium": {
        "Low": 0.20,
        "Medium": 0.60,
        "High": 0.20
    },

    "High": {
        "Low": 0.05,
        "Medium": 0.25,
        "High": 0.70
    }
}

TRAVERSAL_CPT = {

    ("Low", "Low"): {
        "Failure": 0.70,
        "Success": 0.30
    },

    ("Low", "Medium"): {
        "Failure": 0.85,
        "Success": 0.15
    },

    ("Low", "High"): {
        "Failure": 0.95,
        "Success": 0.05
    },

    ("Medium", "Low"): {
        "Failure": 0.30,
        "Success": 0.70
    },

    ("Medium", "Medium"): {
        "Failure": 0.45,
        "Success": 0.55
    },

    ("Medium", "High"): {
        "Failure": 0.70,
        "Success": 0.30
    },

    ("High", "Low"): {
        "Failure": 0.05,
        "Success": 0.95
    },

    ("High", "Medium"): {
        "Failure": 0.15,
        "Success": 0.85
    },

    ("High", "High"): {
        "Failure": 0.40,
        "Success": 0.60
    }
}


def normalize(d):

    total = sum(d.values())

    if total == 0:
        return {
            k: 1 / len(d)
            for k in d
        }

    return {
        k: v / total
        for k, v in d.items()
    }


def safety_distribution(p_safe):

    p_safe = np.clip(p_safe, 0.0, 1.0)

    low = (1 - p_safe) * 0.7
    medium = (1 - p_safe) * 0.3
    high = p_safe

    total = low + medium + high

    return {
        "Low": low / total,
        "Medium": medium / total,
        "High": high / total
    }


def battery_distribution(row):

    low = (
        row["battery_very_low_prob"] +
        row["battery_low_prob"]
    )

    medium = row["battery_moderate_prob"]

    high = (
        row["battery_high_prob"] +
        row["battery_very_high_prob"]
    )

    total = low + medium + high

    return {
        "Low": low / total,
        "Medium": medium / total,
        "High": high / total
    }


def infer_safety(terrain_distribution):

    result = {
        "Low": 0.0,
        "Medium": 0.0,
        "High": 0.0
    }

    for terrain_state, terrain_prob in terrain_distribution.items():

        for safety_state, probability in SAFETY_CPT[terrain_state].items():

            result[safety_state] += (
                terrain_prob * probability
            )

    return normalize(result)


def infer_battery(energy_distribution):

    result = {
        "Low": 0.0,
        "Medium": 0.0,
        "High": 0.0
    }

    for energy_state, energy_prob in energy_distribution.items():

        for battery_state, probability in BATTERY_CPT[energy_state].items():

            result[battery_state] += (
                energy_prob * probability
            )

    return normalize(result)


def infer_traversal(safety_distribution,
                    battery_distribution):

    result = {
        "Failure": 0.0,
        "Success": 0.0
    }

    for safety_state, safety_prob in safety_distribution.items():

        for battery_state, battery_prob in battery_distribution.items():

            joint_probability = (
                safety_prob *
                battery_prob
            )

            for traversal_state, probability in TRAVERSAL_CPT[
                (safety_state, battery_state)
            ].items():

                result[traversal_state] += (
                    joint_probability *
                    probability
                )

    return normalize(result)


def terrain_distribution(row):

    soil = row["soil_prob"]
    bedrock = row["bedrock_prob"]
    sand = row["sand_prob"]
    rock = row["rock_prob"]
    unknown = row["unknown"]

    risk = (
        0.40 * sand +
        0.35 * rock +
        0.15 * bedrock +
        0.10 * unknown
    )

    risk = np.clip(risk, 0, 1)

    low = 1 - risk
    high = risk

    medium = 0.0

    return normalize({
        "Low": low,
        "Medium": medium,
        "High": high
    })


def energy_distribution(row):

    slope = np.clip(
        row["slope_deg"] / 20,
        0,
        1
    )

    roughness = np.clip(
        row["roughness"],
        0,
        1
    )

    slip = np.clip(
        row["wheel_slip"],
        0,
        1
    )

    risk = (
        0.40 * slope +
        0.30 * roughness +
        0.30 * slip
    )

    risk = np.clip(risk, 0, 1)

    return normalize({
        "Low": 1 - risk,
        "Medium": 0.0,
        "High": risk
    })


def run_bayesian_network(row):

    # ResNet → TerrainRisk
    terrain = terrain_distribution(row)

    # TerrainRisk → Safety
    safety = infer_safety(terrain)

    # Sensors → EnergyRisk
    energy = energy_distribution(row)

    # EnergyRisk → BatteryCost
    battery_from_sensors = infer_battery(energy)

    # Jev battery evidence
    battery_from_jev = battery_distribution(row)

    # Combine sensor + Jev battery distributions
    battery = normalize({
        state:
            battery_from_sensors[state] *
            battery_from_jev[state]

        for state in ["Low", "Medium", "High"]
    })

    # Jev safety evidence
    jev_safety = safety_distribution(
        row["jev_safety_probability"]
    )

    # Combine BN safety + Jev safety
    safety = normalize({
        state:
            safety[state] *
            jev_safety[state]

        for state in ["Low", "Medium", "High"]
    })

    # Final inference
    traversal = infer_traversal(
        safety,
        battery
    )

    return {
        "terrain": terrain,
        "safety": safety,
        "battery": battery,
        "traversal": traversal
    }


def add_bn_results(candidate_df):

    candidate_df = candidate_df.copy()

    bn_results = []

    for _, row in candidate_df.iterrows():

        result = run_bayesian_network(row)

        bn_results.append({
            "bn_traversal_success": result["traversal"]["Success"],
            "bn_traversal_failure": result["traversal"]["Failure"]
        })

    bn_df = pd.DataFrame(bn_results)

    candidate_df = pd.concat(
        [
            candidate_df.reset_index(drop=True),
            bn_df
        ],
        axis=1
    )

    return candidate_df
