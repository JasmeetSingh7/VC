import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np



# Initialize an empty list for car names
car_names = []

excel_data = pd.read_excel("C:\\Users\\jasmeet.singh\\Desktop\\AIModelMakeModelCleaned.xlsx", header=None)
car_names = excel_data[0].tolist()
# Iterate over the car names and handle NaN values

# Iterate over the car names and handle NaN values
for car_name in excel_data[0]:
    if isinstance(car_name, str) and not pd.isnull(car_name):
        car_names.append(car_name)
    else:
        car_names.append("")

labeled_examples = [(car_name, 1) for car_name in car_names] + [("example", 0), ("text", 0)]

from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(labeled_examples, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
train_features = vectorizer.fit_transform([example[0] if pd.notnull(example[0]) else "" for example in train_data])
classifier = MultinomialNB()
classifier.fit(train_features, [example[1] for example in train_data])

classifier = MultinomialNB()
classifier.fit(train_features, [example[1] for example in train_data])


from fuzzywuzzy import fuzz, process
import re

def check_for_car_names(classifier, text_file):
    # Read and preprocess the text data
    with open(text_file, 'r', encoding="utf8") as file:
        text = file.read()

    # Extract car names from the excel data
    car_names_lower = [car_name for car_name in car_names]

    # Initialize a list to store the matched car names
    matched_car_names = []

    # Check for exact matches of the entire car name
    for car_name_lower in car_names_lower:
        if car_name_lower in text:
            matched_car_names.append(car_name_lower)

    # Split the text into individual words
    words = re.findall(r'\b\w+\b', text)

    # Iterate over the car names and check for fuzzy matches
    for car_name_lower in car_names_lower:
        ratio = fuzz.token_set_ratio(car_name_lower, text)
        if ratio >= 93:
            matched_car_names.append(car_name_lower)

    return matched_car_names


# Call the function and print the matched car names
matches = check_for_car_names(classifier, "C:/users/jasmeet.singh/desktop/text_file.txt")

if len(matches) > 0:
    
    max_length_car_name = max(matches, key=len)
    print('Match: '+max_length_car_name)
    print("Matched car names:")
    for car_name in matches:
        print(car_name)
else:
    print("No car names found.")












    

