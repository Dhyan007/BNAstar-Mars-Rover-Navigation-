from typesafe_sdk import Noul, Score, TypeSafeClient

client = TypeSafeClient(model="jev-1.13.0")


def get_jev_safety(row):

    state = {
        "mission": {
            "vehicle": "Mars rover",
            "task": "Evaluate a 10 meter candidate terrain segment"
        },

        "terrain_perception": {
            "soil_probability": float(row["soil_prob"]),
            "bedrock_probability": float(row["bedrock_prob"]),
            "sand_probability": float(row["sand_prob"]),
            "big_rock_probability": float(row["rock_prob"]),
            "unknown_probability": float(row["unknown"])
        },

        "rover_sensors": {
            "slope_deg": float(row["slope_deg"]),
            "distance_m": 10.0,
            "roughness": float(row["roughness"]),
            "elevation_change_m": float(row["elevation_change_m"]),
            "speed_mps": float(row["speed_mps"]),
            "wheel_slip": float(row["wheel_slip"])
        }
    }

    response = client.system_one(
        state=state,

        questions={

            # -------------------------
            # 1. SAFETY
            # -------------------------

            "safe": Noul(
                instructions=(
                    "Is this candidate terrain segment safely "
                    "traversable by the Mars rover?"
                ),

                criteria={
                    "true": (
                        "The available terrain and rover sensor evidence "
                        "supports safe traversal of this 10 meter segment "
                        "without a significant terrain-related traversal hazard."
                    ),

                    "false": (
                        "The available terrain and rover sensor evidence "
                        "indicates that traversal presents a significant "
                        "terrain-related hazard, or the evidence is "
                        "insufficient to safely approve traversal."
                    )
                }
            ),

            # -------------------------
            # 2. BATTERY COST
            # -------------------------

            "battery_cost": Score(
                instructions=(
                    "For traversing this 10 meter terrain segment, "
                    "what level of battery energy consumption is expected "
                    "from the Mars rover given the terrain and rover sensor "
                    "conditions?"
                ),

                criteria=[
                    "Very low energy consumption",
                    "Low energy consumption",
                    "Moderate energy consumption",
                    "High energy consumption",
                    "Very high energy consumption"
                ]
            )
        }
    )

    return {
        "safety_probability": response.answers["safe"].noul,

        "battery_cost_score": response.answers["battery_cost"].score,

        "battery_cost_probabilities":
            response.answers["battery_cost"].probabilities
    }


def add_jev_results(candidate_df):
    candidate_df = candidate_df.copy()

    safety_results = []
    battery_results = []

    for _, row in candidate_df.iterrows():

        result = get_jev_safety(row)

        safety_results.append(
            result["safety_probability"]
        )

        battery_results.append(
            result["battery_cost_probabilities"]
        )

    candidate_df["jev_safety_probability"] = safety_results

    candidate_df["battery_very_low_prob"] = [
        x[0] for x in battery_results
    ]

    candidate_df["battery_low_prob"] = [
        x[1] for x in battery_results
    ]

    candidate_df["battery_moderate_prob"] = [
        x[2] for x in battery_results
    ]

    candidate_df["battery_high_prob"] = [
        x[3] for x in battery_results
    ]

    candidate_df["battery_very_high_prob"] = [
        x[4] for x in battery_results
    ]

    return candidate_df
