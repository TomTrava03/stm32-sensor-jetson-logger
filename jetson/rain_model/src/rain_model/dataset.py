"""Dataset utilities for the rain model."""
import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(ENV_PATH)

STATION_LATITUDE = float(os.environ['LATITUDE'])
STATION_LONGITUDE = float(os.environ['LONGITUDE'])
STATION_MODEL = os.environ['WEATHER_ARCHIVE_MODEL']
STATION_TIMEZONE = os.environ['TIMEZONE']

API_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
REQUEST_TIMEOUT_SECONDS = 30

HOURLY_VARIABLES = (
    'temperature_2m', 'relative_humidity_2m',
    'surface_pressure', 'rain',
)


def fetch_weather_data(start_date: str, end_date: str) -> dict[str, Any]:
    params = {
        'start_date': start_date,
        'end_date': end_date,
        'latitude': STATION_LATITUDE,
        'longitude': STATION_LONGITUDE,
        'models': STATION_MODEL,
        'timezone': STATION_TIMEZONE,
        'hourly': ','.join(HOURLY_VARIABLES),
    }

    try:
        response = requests.get(
            API_METEO_URL,
            params=params,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        raise RuntimeError("Weather API returned an HTTP error") from e
    except requests.exceptions.RequestException as e:
        raise RuntimeError("Weather API request failed") from e

    data = response.json()
    if not isinstance(data, dict):
        raise RuntimeError("Weather API response is not a JSON object")

    hourly_data = data.get("hourly")
    if not isinstance(hourly_data, dict):
        raise RuntimeError("Weather API response contains no hourly data")

    timestamps = hourly_data.get("time")
    if not isinstance(timestamps, list) or not timestamps:
        raise RuntimeError(
            "Weather API response contains no hourly timestamps"
        )

    missing_variables = [
        variable
        for variable in HOURLY_VARIABLES
        if variable not in hourly_data
    ]
    if missing_variables:
        missing_names = ", ".join(missing_variables)
        raise RuntimeError(
            "Weather API response is missing hourly variables: "
            f"{missing_names}"
        )

    return data


def save_raw_weather_data(data: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(mode="w", encoding="utf-8") as data_file:
        json.dump(data, data_file)
        data_file.write("\n")


def weather_data_to_dataframe(data: dict[str, Any]) -> pd.DataFrame:
    hourly_data = data.get("hourly")
    dataframe = pd.DataFrame(hourly_data)

    dataframe = dataframe.loc[:, [
        "time",
        "temperature_2m",
        "relative_humidity_2m",
        "surface_pressure",
        "rain",
    ]]

    dataframe.rename(
        columns={
            "time": "timestamp_utc",
            "temperature_2m": "temperature_c",
            "relative_humidity_2m": "relative_humidity_pct",
            "surface_pressure": "surface_pressure_hpa",
            "rain": "rain_mm",
        },
        inplace=True,
    )

    dataframe["timestamp_utc"] = pd.to_datetime(
        dataframe["timestamp_utc"],
        utc=True,
        errors="raise",
    )
    dataframe = dataframe.sort_values("timestamp_utc")

    if dataframe.duplicated(["timestamp_utc"], keep=False).any():
        raise RuntimeError("Found duplicate timestamp")

    return dataframe


def save_processed_weather_data(
    dataframe: pd.DataFrame,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataframe.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )


def build_weather_dataset(
    start_date: str,
    end_date: str,
    raw_output_path: Path,
    processed_output_path: Path,
) -> pd.DataFrame:
    weather_data: dict[str, Any] = fetch_weather_data(
        start_date=start_date,
        end_date=end_date,
    )
    save_raw_weather_data(data=weather_data, output_path=raw_output_path)

    weather_dataframe = weather_data_to_dataframe(data=weather_data)
    save_processed_weather_data(
        dataframe=weather_dataframe,
        output_path=processed_output_path,
    )

    return weather_dataframe


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--processed-output", type=Path, required=True)

    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_arguments()

    dataframe = build_weather_dataset(
        start_date=args.start_date,
        end_date=args.end_date,
        raw_output_path=args.raw_output,
        processed_output_path=args.processed_output,
    )

    print(f"Saved {len(dataframe)} hourly weather records")


if __name__ == "__main__":
    main()