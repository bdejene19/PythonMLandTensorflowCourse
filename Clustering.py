# Clustering is the first unsupervised algorithm we've looked at
# is very powerful
# clustering --> used only when having lots of input information (features) but have no output information (i.e. labels)
# clustering finds like data points and finds the locations of those data points


'''
we will be doing be doing clustering via K-means
# first we randomly pick points to place centroids
# Centroids: where our current cluster is defined
# random K-centroids are placed where ever on the graph of data points
# each data point is assigned to a cluster by distance --> find the uclivian or manhattan distance
# assign the data point to the closest centroid ==> we then do this for every single data point
Then you move the centroid to the middle of all its data points (its center of mass)

Repeat the process process with the re-evaluated placement of the centroids
Keep doing this where none of the data points are changing which centroid they belong to
    i.e. centroids are as central as possible ==> now have cluster with centroids resembling center of mass

When we have have a new data point --> we see the distance to the centroids for each cluster
    depending on distance, you assign data point to a particular cluster that has an assigned label
'''
