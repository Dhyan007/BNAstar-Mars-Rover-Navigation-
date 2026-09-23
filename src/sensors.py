import numpy as np
import pandas as pd


def add_sensor_data(candidate_df, seed=42):

    np.random.seed(seed)

    candidate_df = candidate_df.copy()

    sensor_columns = [
        "slope_deg",
        "distance_m",
        "roughness",
        "elevation_change_m",
        "speed_mps",
        "wheel_slip"
    ]

    # Remove old sensor columns if they already exist
    candidate_df = candidate_df.drop(
        columns=sensor_columns,
        errors="ignore"
    )

    sensor_data = []

    for _, row in candidate_df.iterrows():

        bedrock = row["bedrock_prob"]
        sand = row["sand_prob"]
        big_rock = row["rock_prob"]

        slope_deg = np.random.uniform(0, 15)

        slope_deg += (
            sand * np.random.uniform(0, 5)
            + big_rock * np.random.uniform(0, 4)
        )

        slope_deg = np.clip(slope_deg, 0, 20)

        roughness = (
            0.10
            + 0.35 * bedrock
            + 0.60 * big_rock
            + np.random.normal(0, 0.05)
        )

        roughness = np.clip(roughness, 0, 1)

        elevation_change_m = (
            slope_deg * np.random.uniform(0.05, 0.20)
            + np.random.normal(0, 0.2)
        )

        elevation_change_m = max(
            0,
            elevation_change_m
        )

        distance_m = 10.0

        base_speed = np.random.uniform(0.05, 0.15)

        speed_reduction = (
            0.04 * sand
            + 0.03 * big_rock
            + 0.02 * (slope_deg / 20)
        )

        speed_mps = np.clip(
            base_speed - speed_reduction,
            0.02,
            0.20
        )

        wheel_slip = (
            0.02
            + 0.35 * sand
            + 0.10 * (slope_deg / 20)
            + 0.10 * roughness
            + np.random.normal(0, 0.02)
        )

        wheel_slip = np.clip(
            wheel_slip,
            0,
            1
        )

        sensor_data.append({
            "slope_deg": slope_deg,
            "distance_m": distance_m,
            "roughness": roughness,
            "elevation_change_m": elevation_change_m,
            "speed_mps": speed_mps,
            "wheel_slip": wheel_slip
        })

    sensor_df = pd.DataFrame(sensor_data)

    candidate_df = pd.concat(
        [
            candidate_df.reset_index(drop=True),
            sensor_df
        ],
        axis=1
    )

    return candidate_df
