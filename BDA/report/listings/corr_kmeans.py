import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

np.random.seed(42)

from platform import system

source = ("data\\" if system() == "Windows" else "data/") + "diamonds.csv"

N = 2000

df = pd.read_csv(source).sample(n=N, random_state=42)

print(df.shape)
df.head(8)

df["cut_ord"] = df["cut"].map(
    {"Fair": 0.0, "Good": 1.0, "Very Good": 2.0, "Premium": 3.0, "Ideal": 4.0}
)
df["clarity_ord"] = df["clarity"].map(
    {
        "I1": 0.0,
        "SI2": 1.0,
        "SI1": 2.0,
        "VS2": 3.0,
        "VS1": 4.0,
        "VVS2": 5.0,
        "VVS1": 6.0,
        "IF": 7.0,
    }
)

df["color_ord"] = df["color"].map(
    {
        "J": 0.0,
        "I": 1.0,
        "H": 2.0,
        "G": 3.0,
        "F": 4.0,
        "E": 5.0,
        "D": 6.0,
    }
)
df.head()

df = df.drop(["cut", "clarity", "color"], axis=1)
# df = pd.get_dummies(df, columns=["color"])


# # Assignment 1. Correlation coefficient
# 
# With the data preprocessed, we have 1 categorical feature and 9 numerical features. In this task a linear regression should be made with a pair of features and several metrics calculated on its prediction results.

df.head()

df.corr()


# Let's see the pairvise scatterplots of all the features in our data.

pplot = sns.pairplot(df, diag_kind="kde")
pplot.savefig("media/pairplot.png")
plt.show()

with sns.axes_style("whitegrid"):
    sns.scatterplot(df, x="carat", y="price", alpha=0.5)
plt.show()

from sklearn.linear_model import LinearRegression


x = df["carat"].to_numpy().reshape(-1, 1)
y = df["price"].to_numpy().reshape(-1, 1)

reg = LinearRegression().fit(x, y)
y_pred = reg.predict(x).reshape(-1, 1)

with sns.axes_style("whitegrid"):
    sns.scatterplot(df, x="carat", y="price", alpha=0.2)
    sns.lineplot(df, x="carat", y=y_pred.flatten(), color="r")
plt.show()

corr = df.corr()["carat"]["price"]
corr

ss_res = ((y - y_pred) ** 2).sum()
ss_tot = ((y - np.mean(y)) ** 2).sum()

determinacy = 1.0 - ss_res / ss_tot
determinacy


# Let's check whether the calculated coefficient of determination is correct. Since we are using least-squares linear regression, it should be equal to the correlation coefficient squared.

np.abs(determinacy - corr**2) < 1e-9


# Linear regression coefficients

a = corr * np.std(y) / np.std(x)
b = np.mean(y) - a * np.mean(x)


# Let's check the relative deviations of predicted values from true values and vise-versa for a set of 4 randomly selected instances.

idx = np.random.randint(0, y.shape[0] - 1, size=4)

x_cut = x[idx]
y_cut = y[idx]

dy_cut = y_cut - (a * x_cut + b)

dev_from_true = 100 * dy_cut / y_cut
dev_from_pred = 100 * dy_cut / (a * x_cut + b)

x_cut

y_cut

dev_from_true

dev_from_pred


# Now, let's do the same for each instance in the dataset

dy = y - (a * x + b)

tot_dev_from_true = np.abs(100 * dy / y).mean()
tot_dev_from_pred = np.abs(100 * dy / (a * x + b)).mean()

print(
    f"MAPE(relative to true values): {tot_dev_from_true:0.4f}%\nMAPE(relative to predicted values): {tot_dev_from_pred:0.4f}%"
)


# Now let's try polynomial regression.

x_stacked = np.hstack([np.ones_like(x), x, x**2, x**3])

reg = LinearRegression().fit(x_stacked, y)

with sns.axes_style("whitegrid"):
    sns.scatterplot(df, x="carat", y="price", alpha=0.2)
    sns.lineplot(df, x="carat", y=reg.predict(x_stacked).flatten(), color="r")
plt.show()

y_stacked = reg.predict(x_stacked).reshape(-1, 1)
dy_stacked = y - y_stacked


tot_dev_from_true_stacked = np.abs(100 * dy / y).mean()
tot_dev_from_pred_stacked = np.abs(100 * dy / y_stacked).mean()

print(
    f"MAPE(relative to true values): {tot_dev_from_true_stacked:0.4f}%\nMAPE(relative to predicted values): {tot_dev_from_pred_stacked:0.4f}%"
)

x = df["x"].to_numpy().reshape(-1, 1)
y = df["carat"].to_numpy().reshape(-1, 1)

x_stacked = np.hstack([np.ones_like(x), x, x**2, x**3])

reg = LinearRegression().fit(x_stacked, y)

with sns.axes_style("whitegrid"):
    sns.scatterplot(df, x="x", y="carat", alpha=0.2)
    sns.lineplot(df, x="x", y=reg.predict(x_stacked).flatten(), color="r")
plt.show()


# # Assignment 3. K-means clustering
# 
# For the K-means clustering we will use the following features:  `depth`, `table`, `carat` and `price`. First, we standardize our features:

from sklearn.preprocessing import StandardScaler

frame = df[["depth", "table", "carat", "price"]]

frame_std = pd.DataFrame(
    StandardScaler().fit(frame).transform(frame),
    columns=["depth", "table", "carat", "price"],
)
frame.head()

frame_std.head()

from sklearn.cluster import KMeans

state = np.random.RandomState(seed=42)

grand_mean = frame.to_numpy().mean(axis=0)
print(f"Grand mean: {grand_mean}")

for n in [4, 7]:
    intertiae = []
    kmeans = []
    for i in range(12):
        kmeans += [KMeans(n_clusters=n, random_state=state, n_init=1).fit(frame_std)]
        intertiae += [kmeans[i].inertia_]

    best_idx = np.argmin(np.array(intertiae[1:]))
    clusters = kmeans[best_idx + 1]

    print(
        f"\n\nn_clusters: {n}\nall intertia values:\n\n{intertiae}\n\nbest k-means inertia(run #{best_idx+1}): {clusters.inertia_:0.4f}\n"
    )

    table_vals = pd.DataFrame(
        [[1, 2, 3, 4]], columns=["depth", "table", "carat", "price"]
    )

    for j in range(clusters.n_clusters):
        cluster = frame[clusters.labels_ == j]
        cluster_center = cluster.to_numpy().mean(axis=0)
        new_row = pd.DataFrame(
            [
                cluster_center,
                cluster_center - grand_mean,
                (cluster_center - grand_mean) / grand_mean,
            ],
            columns=["depth", "table", "carat", "price"],
            index=[j * 3 + k for k in range(3)],
        )
        table_vals = pd.concat([table_vals, new_row])
        print(
            f"cluster #{j+1}.\n# of elements: {cluster.shape[0]}\ncenter: {cluster_center}\ngrand mean deviation:{cluster_center - grand_mean}\nrelative grand mean deviation: {(cluster_center - grand_mean)/grand_mean}\n"
        )
    table_vals.to_csv(f"{n}_clusters.csv")
