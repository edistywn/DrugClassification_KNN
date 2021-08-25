from collections import Counter
import math

def knn(data, query, k, distance_fn, choice_fn):

	neighbor_distances_and_indices=[]

	for index, example in enumerate(data):

		distance = distance_fn(example[:-1], query)

		neighbor_distances_and_indices.append((distance, index))

	sorted_neighbor_distance_and_indices = sorted(neighbor_distances_and_indices)

	k_nearest_distances_and_indices = sorted_neighbor_distance_and_indices[:k]

	k_nearest_labels=[data[i][-1] for distance, i in k_nearest_distances_and_indices]

	return k_nearest_labels, choice_fn(k_nearest_labels)


def mean(labels):

	return sum(labels)/len(labels)

def mode(labels):

	return Counter(labels).most_common(1)[0][0]

def euclidean_dist(point1, point2):

	sum_squared_distance=0

	for i in range(len(point1)):
		sum_squared_distance += math.pow(point1[i] - point2[i], 2)

	return math.sqrt(sum_squared_distance)


def main():

	reg_data=[
	[150, 60],
	[160, 65],
	[165, 67],
	[155, 55],
	[170, 68],
	[180, 70]
	]

	query_data=[158]

	reg_knn, reg_pred = knn(reg_data, query_data, k=3, distance_fn=euclidean_dist, choice_fn=mean )

	print(reg_knn)
	print(reg_pred)

	class_data=[
	[22, 1],
	[23, 1],
	[24, 1],
	[25, 2],
	[26, 2],
	[27, 2],

	]

	class_query=[30]

	class_knn, class_pred=knn(class_data, class_query, k=3, distance_fn=euclidean_dist, choice_fn=mode)

	print(class_knn)
	print(class_pred)

if __name__ == '__main__':
	main()


