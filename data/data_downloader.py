import os
import pandas as pd
from data_utils import download_country_data


def get_categories(name_p):
    return [f'Category:21st-century_{name_p}_people',
            f'Category:20th-century_{name_p}_people',
            f'Category:{name_p}_people_by_occupation']


if __name__ == "__main__":
    data = pd.DataFrame(columns=['name', 'nationality'])

    size_per_search = 10
    max_per_category = 1

    # Europe
    data = download_country_data(data, get_categories('German'), 'DE', size_per_search, max_per_category)
    data = download_country_data(data, get_categories('French'), 'FR', size_per_search, max_per_category)
    data = download_country_data(data, get_categories('Italian'), 'IT', size_per_search, max_per_category)
    data = download_country_data(data, get_categories('Spanish'), 'ES', size_per_search, max_per_category)
    data = download_country_data(data, get_categories('Polish'), 'PL', size_per_search, max_per_category)
    data = download_country_data(data, get_categories('British'), 'GB', size_per_search, max_per_category)
    data = download_country_data(data, get_categories('Dutch'), 'NL', size_per_search, max_per_category)
    data = download_country_data(data, get_categories('Belgian'), 'BE', size_per_search, max_per_category)
    data = download_country_data(data, get_categories('Portuguese'), 'PT', size_per_search, max_per_category)
    data = download_country_data(data, get_categories('Irish'), 'IE', size_per_search, max_per_category)
    data = download_country_data(data, get_categories('Norwegian'), 'NO', size_per_search, max_per_category)
    data = download_country_data(data, get_categories('Swedish'), 'SE', size_per_search, max_per_category)
    data = download_country_data(data, get_categories('Danish'), 'DK', size_per_search, max_per_category)
    data = download_country_data(data, get_categories('Finnish'), 'FI', size_per_search, max_per_category)
    data = download_country_data(data, get_categories('Czech'), 'CZ', size_per_search, max_per_category)
    data = download_country_data(data, get_categories('Slovak'), 'SK', size_per_search, max_per_category)
    data = download_country_data(data, get_categories('Austrian'), 'AT', size_per_search, max_per_category)
    data = download_country_data(data, get_categories('Hungary'), 'HU', size_per_search, max_per_category)
    data = download_country_data(data, get_categories('Romanian'), 'RO', size_per_search, max_per_category)
    data = download_country_data(data, get_categories('Bulgarian'), 'BG', size_per_search, max_per_category)
    data = download_country_data(data, get_categories('Greek'), 'GK', size_per_search, max_per_category)
    data = download_country_data(data, get_categories('Russian'), 'RU', size_per_search, max_per_category)
    data = download_country_data(data, get_categories('Ukrainian'), 'UA', size_per_search, max_per_category)
    data = download_country_data(data, get_categories('Belarusian'), 'BY', size_per_search, max_per_category)

    if not os.path.exists('data/csv/'):
        os.makedirs('data/csv/')
    filename = f'data/csv/nationalities_{size_per_search}_{max_per_category}.csv'
    data.to_csv(filename) # save to data/csv/
    data.to_csv('nationalities.csv') # save to main folder
    print(f'Data downloaded and saved to {filename}')
