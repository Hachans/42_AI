import numpy as np
import pandas as pd
from predict import extract_values

# https://towardsdatascience.com/r-squared-recipe-5814995fa39a
def r2_score(df, theta0, theta1):
    x = df["km"]

    line1 = (theta0 + (theta1 * x))
    line1_diff = line1 - df.price
    line1_sum = 0
    for i in line1_diff:
        line1_sum += (i*i)

    line2 = np.full(len(x), [df.price.mean()])
    line2_diff = line2 - df.price
    line2_sum = 0
    for j in line2_diff:
        line2_sum += (j*j)

    return (line2_sum - line1_sum)/line2_sum

if __name__ == "__main__":
    df = pd.read_csv('data/data.csv')

    theta0, theta1 = extract_values()
    r2 = r2_score(df, theta0, theta1)
    print(f"R squared score of the model is: {r2}")