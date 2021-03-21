import pandas as pd
import pprint
from sklearn.cluster import KMeans

n_cluster = 10

names = pd.read_csv("monster.txt", sep=',', na_values=".")

df = pd.read_csv("monster.txt", sep=',', na_values=".")
df = df.drop('name',axis=1).drop('color',axis=1)

kmeans_model = KMeans(n_clusters=n_cluster,random_state=10).fit(df.iloc[:, 1:])
labels = kmeans_model.labels_

monster_type = []
for i in range(n_cluster):
    monster_type.append([])

n = 0
for i in labels.tolist():
    name = names.iloc[n]['name']
    monster_type[i].append(name)
    n = n + 1

pprint.pprint(monster_type)

# https://www.color-site.com/separate_hues
color_codes = {
    0:'#BA7836',
    1:'#ADBA36',
    2:'#5EBA36',
    3:'#36BA5E',
    4:'#36BAAD',
    5:'#3678BA',
    6:'#4336BA',
    7:'#9236BA',
    8:'#BA3692',
    9:'#BA3643',
}
colors = [color_codes[x] for x in labels]

import matplotlib.pyplot as plt
from pandas import plotting

plotting.scatter_matrix(
    df[df.columns[1:]],
    figsize=(6,6),
    color=colors,
    alpha=0.8, 
    diagonal='kde'
)
plt.show()
