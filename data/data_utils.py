import requests
import pandas as pd


url = "https://en.wikipedia.org/w/api.php"

def generate_names(category_title, name_list, visited_dict, max_length=float('inf'), max_per_category=float('inf')):

    visited_dict[category_title] = True

    params = {
        "action": "query",
        "cmtitle": category_title,
        "cmlimit": "500",
        "list": "categorymembers",
        "format": "json"
    }

    category_counter = 0

    while True:
        req = requests.get(url=url, params=params)
        json = req.json()


        for it in json['query']['categorymembers']:

            if len(name_list) < max_length:
                if it['ns'] == 0 and category_counter < max_per_category:
                    if "List of" not in it['title'] and it['title'] not in visited_dict:
                        name_list.append(it['title'])
                        visited_dict[it['title']] = True
                        category_counter += 1
                elif it['ns'] == 14:
                    if it['title'] not in visited_dict:
                        generate_names(it['title'], name_list, visited_dict, max_length, max_per_category)

        if "continue" in json:
            params["cmcontinue"] = json["continue"]["cmcontinue"]
        else:
            break


def generate_names_list(category_title, visited_dict, size_per_search, max_per_category):
    name_list_temp = []
    generate_names(category_title, name_list_temp, visited_dict, size_per_search, max_per_category)
    return name_list_temp


def full_list(categories, size_per_search, max_per_category = 100):
    name_list = []
    visited_dict = {}

    for category in categories:
        name_list.extend(generate_names_list(category, visited_dict, size_per_search, max_per_category))

    return name_list


def download_country_data(data, categories, code, size_per_search, max_per_category):
    name_list = full_list(categories, size_per_search, max_per_category)
    return pd.concat([data, pd.DataFrame({'name': name_list, 'nationality': code})], ignore_index=True)

