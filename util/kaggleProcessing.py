import csv
import sys

filename = 'news_data/fake.csv'

TITLE = 4
TEXT = 5

csv.field_size_limit(sys.maxsize)

def read_kaggle_dataset():

    with open(filename, 'r') as f, open("news_data/kaggle.txt", "w") as fout:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            sample = row[TITLE] + " " + row[TEXT] + "\n"
            fout.write(sample)
