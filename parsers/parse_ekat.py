import cianparser
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def main():
    ekat_parser = cianparser.CianParser(location='Екатеринбург')
    for i in range(5):
        data = ekat_parser.get_flats(deal_type='sale', rooms=(i+1), with_saving_csv=True)
        print(len(data))
        print(data[0])

    # ds = pd.read_csv('cian_flat_sale_1_100_ekaterinburg_10_Apr_2025_02_13_40_633322.csv', delimiter=';')
    # print(ds.columns)
    # print(ds.head())



if __name__ == '__main__':
    main()
