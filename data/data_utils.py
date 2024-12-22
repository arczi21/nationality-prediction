import tqdm
import requests
import pandas as pd

def collect(name_p, code, list_len=5000):

    url = "https://en.wikipedia.org/w/api.php"

    category_params = {
        "action": "query",
        "cmtitle": None,
        "cmlimit": "max",
        "list": "categorymembers",
        "format": "json",
        "cmcontinue": None,
    }

    #start_category = f'Category:21st-century_{name_p}_people'
    start_category = f'Category:{name_p}_people'

    visited_dict = {}
    categories = [(start_category, 0)]
    people_list = []

    print(f'{code}:')

    with tqdm.tqdm(total=list_len) as progress:

      while len(people_list) < list_len:
        if len(categories) > 0:
            category_name, tier = categories.pop()
            category_params["cmtitle"] = category_name
            category_params["cmcontinue"] = None
            visited_dict[category_name] = True

            while True:

                response = requests.get(url, category_params)
                data = response.json()

                for query in data['query']['categorymembers']:
                    if query['title'] not in visited_dict and len(people_list) < list_len:
                        visited_dict[query['title']] = True
                        if query['ns'] == 0:
                            people_list.append(query['title'])
                            progress.update(1)
                        elif query['ns'] == 14:
                            if name_p in query['title']:
                                categories.append((query['title'], tier+1))

                if "continue" in data:
                    category_params["cmcontinue"] = data["continue"]["cmcontinue"]
                else:
                    break
        else:
            break
    return people_list


def download_country_data(data, name_p, code, size_per_search):
    name_list = collect(name_p, code, size_per_search)
    return pd.concat([data, pd.DataFrame({'name': name_list, 'nationality': code})], ignore_index=True)

