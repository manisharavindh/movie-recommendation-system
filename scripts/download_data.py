import urllib.request
import zipfile
import os

def download_data():
    url = 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip'

    print('Downloading MovieLens Dataset...')
    urllib.request.urlretrieve(url, 'ml-latest-small.zip')

    with zipfile.ZipFile('ml-latest-small.zip', 'r') as zip_ref:
        zip_ref.extractall('data/raw/')

    os.remove('ml-latest-small.zip')
    print('Data Downloaded to data/raw/')

if __name__ == '__main__':
    download_data()