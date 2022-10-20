import pandas as pd

genre = pd.read_csv("./data/genre.csv")

genre.head()

genre["files"] = genre["files"].map(lambda x: x[11:-2])
genre["labels"] = genre["labels"].map(lambda x: x[11:])

label_map = {"blues": 1, "classical": 2, "country": 3, "rock": 4, "jazz": 5}

genre["y"] = genre["labels"].map(label_map)

genre.head()

genre.to_csv("./data/genre_clean.csv", index=False)

mel_specs = pd.read_csv("./data/genre_mel_specs.csv")

mel_specs = mel_specs.rename(columns={"84480": "labels"})
mel_specs["y"] = mel_specs["labels"].map(label_map)

mel_specs.to_csv("./data/genre_mel_specs_clean.csv", index=False)
