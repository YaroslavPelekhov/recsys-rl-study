import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def toy_data():
    rows = []
    for day in range(0, 71, 7):
        for user in range(18):
            rows.append(
                (user, (user + day // 7) % 30, pd.Timestamp("2021-01-01") + pd.Timedelta(days=day))
            )
    events = pd.DataFrame(rows, columns=["user_id", "item_id", "timestamp"])
    items = pd.DataFrame(
        {
            "item_id": np.arange(30),
            "genres": ["drama, comedy" if i % 2 else "comedy" for i in range(30)],
            "directors": [f"director{i % 5}" for i in range(30)],
            "countries": ["RU"] * 30,
            "content_type": ["film"] * 30,
            "age_rating": [16] * 30,
            "for_kids": [0] * 30,
            "release_year": [2020] * 30,
        }
    ).set_index("item_id")
    users = pd.DataFrame(
        {
            "user_id": np.arange(18),
            "age": ["25_34"] * 18,
            "income": ["60_90"] * 18,
            "sex": ["unknown"] * 18,
            "kids_flg": [0] * 18,
        }
    ).set_index("user_id")
    return events, items, users
