import json
import pandas as pd

def extract_values():
    with open('data/args.json', 'r') as json_file:
        data = json.load(json_file)
    theta0 = data['theta0']
    theta1 = data['theta1']

    return theta0, theta1

if __name__ == "__main__":
    mileage = float(input('Enter mileage to predict: '))
    theta0, theta1 = extract_values()
    print(f'price prediction: {theta0 + (theta1 * mileage)}')
