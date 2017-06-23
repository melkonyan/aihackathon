import csv
import sys

filename = '../data/fake.csv'

TITLE = 4
TEXT = 5

csv.field_size_limit(sys.maxsize)

output = open("../data/kaggle.txt", "w", encoding='utf-8')

with open(filename, 'r', encoding='utf-8') as f:
    dic = {}
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        output.write(row[TITLE] + " " + row[TEXT] + "\n")


output.close()
