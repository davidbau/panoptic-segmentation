import csv
import json

with open('objects.csv') as file, open('categories.json', 'w') as outputfile:
    reader = csv.reader(file)
    categories = []
    first = True
    for row in reader:
        if first:
            first = False
            continue
        categories.append(row[0])

    print(len(categories))

    json.dump(sorted(categories), outputfile)
