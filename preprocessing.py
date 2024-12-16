import re


def preprocessing(data):

    # Removing brackets in name, for example: "Jan Kowalski (blacksmith)" -> "Jan Kowalski"
    data.name = data.name.apply(lambda x: re.sub(r' \(.*?\)', '', x))

    data = data.reset_index(drop=True)

    return data