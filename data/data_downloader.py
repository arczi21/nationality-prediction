import os
import pandas as pd
from data_utils import download_country_data


def get_categories(name_p):
    return [f'Category:21st-century_{name_p}_people',
            f'Category:20th-century_{name_p}_people',
            f'Category:{name_p}_people_by_occupation']


if __name__ == "__main__":
    data = pd.DataFrame(columns=['name', 'nationality'])

    size_per_search = 20000

    # Europe
    data = download_country_data(data, 'German', 'DE', size_per_search)
    data = download_country_data(data, 'French', 'FR', size_per_search)
    data = download_country_data(data, 'Italian', 'IT', size_per_search)
    data = download_country_data(data, 'Spanish', 'ES', size_per_search)
    data = download_country_data(data, 'Polish', 'PL', size_per_search)
    data = download_country_data(data, 'British', 'GB', size_per_search)
    data = download_country_data(data, 'Dutch', 'NL', size_per_search)
    data = download_country_data(data, 'Belgian', 'BE', size_per_search)
    data = download_country_data(data, 'Portuguese', 'PT', size_per_search)
    data = download_country_data(data, 'Irish', 'IE', size_per_search)
    data = download_country_data(data, 'Norwegian', 'NO', size_per_search)
    data = download_country_data(data, 'Swedish', 'SE', size_per_search)
    data = download_country_data(data, 'Danish', 'DK', size_per_search)
    data = download_country_data(data, 'Finnish', 'FI', size_per_search)
    data = download_country_data(data, 'Czech', 'CZ', size_per_search)
    data = download_country_data(data, 'Slovak', 'SK', size_per_search)
    data = download_country_data(data, 'Austrian', 'AT', size_per_search)
    data = download_country_data(data, 'Hungarian', 'HU', size_per_search)
    data = download_country_data(data, 'Romanian', 'RO', size_per_search)
    data = download_country_data(data, 'Bulgarian', 'BG', size_per_search)
    data = download_country_data(data, 'Greek', 'GK', size_per_search)
    data = download_country_data(data, 'Russian', 'RU', size_per_search)
    data = download_country_data(data, 'Ukrainian', 'UA', size_per_search)
    data = download_country_data(data, 'Belarusian', 'BY', size_per_search)

    if not os.path.exists('data/csv/'):
        os.makedirs('data/csv/')
    filename = f'data/csv/nationalities_{size_per_search}.csv'
    data.to_csv(filename) # save to data/csv/
    data.to_csv('nationalities.csv') # save to main folder
    print(f'Data downloaded and saved to {filename}')
