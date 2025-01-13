import tqdm
import requests
import time
import pandas as pd

def collect(name_p, code, list_len=5000, time_limit=600):

    url = "https://en.wikipedia.org/w/api.php"

    category_params = {
        "action": "query",
        "cmtitle": None,
        "cmlimit": "max",
        "list": "categorymembers",
        "format": "json",
        "cmcontinue": None,
    }

    params = {
        "action": "query",
        "format": "json",
        "titles": None,
        "prop": "revisions",
        "rvprop": "content",
    }

    #start_category = f'Category:21st-century_{name_p}_people'
    start_category = f'Category:{name_p}_people'

    visited_dict = {}
    categories = [(start_category, 0)]
    people_query = []
    people_list = []

    print(f'{code}:')

    with tqdm.tqdm(total=list_len) as progress:

        start_time = time.time()

        while len(people_list) < list_len and time.time() - start_time < time_limit:
            if len(people_query) >= 50:
                params['titles'] = "|".join(people_query[0:50])
                people_query = people_query[50:]
                response = requests.get(url, params)
                data = response.json()

                for key in data['query']['pages']:
                    page = data['query']['pages'][key]
                    if '| name' in page['revisions'][0]['*'] and '| birth_place' in page['revisions'][0]['*']:
                        people_list.append(page['title'])
                        progress.update(1)

                time.sleep(0)


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
                                people_query.append(query['title'])
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