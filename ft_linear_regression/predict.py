import json
import pandas as pd

mileage = float(input('Enter mileage to predict: '))

def extract_values():
    with open('args.json', 'r') as json_file:
        data = json.load(json_file)
    theta0 = data['theta0']
    theta1 = data['theta1']

    return theta0, theta1


def main():
    theta0, theta1 = extract_values()
    print(f'price prediction: {theta1 + theta0 * mileage}')


if __name__ == "__main__":
    main()
