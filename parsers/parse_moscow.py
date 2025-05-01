import cianparser
import pandas as pd


def main():
    msk_parser = cianparser.CianParser(location='Москва')
    for i in range(5):
        data = msk_parser.get_flats(deal_type='sale', rooms=(i + 1), with_saving_csv=True)
        print(len(data))
        print(data[0])

    # ds = pd.read_csv('cian_flat_sale_1_100_ekaterinburg_10_Apr_2025_02_13_40_633322.csv', delimiter=';')
    # print(ds.columns)
    # print(ds.head())
    
if __name__ == '__main__':
    main()
