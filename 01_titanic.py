import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def task_1(csv):
    labels = []
    men_means = []
    women_means = []
    parsed_data = {}
    data = pd.read_csv(csv)
    genders = data.Sex
    classes = data.Pclass
    survived = data.Survived
    for index, passenger_class in enumerate(classes):
        if passenger_class not in parsed_data.keys():
            parsed_data[passenger_class] = {
                "male": 0,
                "female": 0
            }
            if survived[index] == 1:
                if genders[index] == "male":
                    parsed_data[passenger_class]["male"] += 1
                else:
                    parsed_data[passenger_class]["female"] += 1
        else:
            if survived[index] == 1:
                if genders[index] == "male":
                    parsed_data[passenger_class]["male"] += 1
                else:
                    parsed_data[passenger_class]["female"] += 1
    for key in parsed_data.keys():
        labels.append(key)
        men_means.append(parsed_data[key]["male"])
        women_means.append(parsed_data[key]["female"])
        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width / 2, men_means, width, label="Men")
        rects2 = ax.bar(x + width / 2, women_means, width, label="Women")

        ax.set_ylabel("Survived")
        ax.set_title("Survived by class and gender")
        ax.set_xticks(x, labels)
        ax.legend()

        ax.bar_label(rects1, padding=3)
        ax.bar_label(rects2, padding=3)

        fig.tight_layout()

        plt.show()


def task_2():
    labels = []
    total_passengers = []
    data = pd.read_csv("titanic.csv")
    classes = data.Pclass
    all_classes = data["Pclass"].unique()
    results = [(i, classes[classes == i].count()) for i in all_classes]
    for result in results:
        labels.append(result[0])
        total_passengers.append(result[1])
    _, ax1 = plt.subplots()
    ax1.pie(total_passengers, labels=labels, autopct="%1.1f%%",
            shadow=True, startangle=90)
    ax1.axis("equal")

    plt.show()


def task_3():
    data = pd.read_csv("titanic.csv")
    data.hist(column="Age")
    plt.xlabel("Amount of people")
    plt.ylabel("Age")
    plt.show()


if __name__ == "__main__":
    # task_1("titanic.csv") # distribution by survived
    # task_2() # classes of passengers
    # task_3() #distribution by age
    pass
