#Run Packages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score
from itertools import product

#generating samples 
np.random.seed(42)

cluster1 = np.random.multivariate_normal([1, 2], [[0.1, 0.05], [0.05, 0.2]], 1000)
cluster2 = np.random.multivariate_normal([0, 0], [[0.3, -0.1], [-0.1, 0.2]], 500)
cluster3 = np.random.multivariate_normal([-2, 3], [[1.5, 0], [0, 1.5]], 1500)

data = np.vstack((cluster1, cluster2, cluster3))
true_label = np.hstack((np.zeros(1000), np.ones(500), np.full(1500, 2)))

#Applying DBscan eps and min_samples values(testing) 
for i in range(1, 11):
    eps = i * 0.1
    min_samples = i
    dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
    #print(f"(eps={eps}, min_samples={min_samples}):", set(dbscan.labels_))

#modify loop to find the best parameters 
best_score = -1
best_params = None

for eps in np.arange(0.1,1.1,0.1):
    for min_samples in range(1,11):
      dbscan = DBSCAN(eps=eps, min_samples=min_samples)
      y_pred = dbscan.fit_predict(data)
        
        
    if len(np.unique(y_pred)) == 1:
            continue
            
    score = adjusted_rand_score(true_label, y_pred)
        
    if score > best_score:
            best_score = score
            best_params = (eps, min_samples)
            best_labels = y_pred

# Print best parameters and score
print(f"Best parameters: eps={best_params[0]:.1f}, min_samples={best_params[1]}")
print(f"Best ARI: {best_score:.3f}")
print(best_labels)

#Dbscan with best parameters (eps= 0.2 and min_samples = 10)
best_eps = 0.2
best_min_samples = 10
dbscan_best_parameter = DBSCAN(eps=best_eps, min_samples=best_min_samples)
labels_pred = dbscan_best_parameter.fit_predict(data)

#plot with px
df = pd.DataFrame(data, columns=["X", "Y"])
df["Cluster"] = labels_pred.astype(str) 

# Mark outliers
df.loc[df["Cluster"] == "-1", "Cluster"] = "Outlier"

fig = px.scatter(df, x="X", y="Y", color="Cluster",
                 title=f"DBSCAN Clustering (eps={best_eps}, min_samples={best_min_samples})",
                 labels={"X": "Feature 1", "Y": "Feature 2", "Cluster": "Cluster ID"},
                 color_discrete_sequence=px.colors.qualitative.Set1)
fig.show()