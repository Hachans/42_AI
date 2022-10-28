import pandas as pd
import json
import numpy as np

def save_values(theta0, theta1):
    with open("args.json", "r") as f:
        file_data = json.load(f)
    file_data["theta0"] = theta0
    file_data["theta1"] = theta1
    with open("args.json", "w") as f:
        json.dump(file_data, f)

def linear_regression(theta0, theta1, df, l):
    theta0_grad = 0.0
    theta1_grad = 0.0

    n = len(df)

    for i in range(n):
        x = df.km.iloc[i]
        y = df.price.iloc[i]

        theta0_grad += -(2/n) * x * (y - (theta0 * x + theta1))
        theta1_grad += -(2/n) * (y - (theta0 * x + theta1))

    theta0 = theta0 - theta0_grad * l
    theta1 = theta1 - theta1_grad * l

    return theta0, theta1

def main():
    l = 0.01
    epochs = 4000
    theta0 = 0
    theta1 = 0

    df = pd.read_csv('data.csv')

    km_max = df.km.max()
    price_max = df.price.max()

    df.km = df.km / km_max
    df.price = df.price / price_max


    for i in range(epochs):
        if i % 100 == 0:
            print(f"epoch: {i}")
        theta0, theta1 = linear_regression(theta0, theta1, df, l)

    theta0 *= price_max/km_max
    theta1 *= price_max 

    print(theta0, theta1)
    save_values(theta0, theta1)


if __name__ == "__main__":
    main()

 
