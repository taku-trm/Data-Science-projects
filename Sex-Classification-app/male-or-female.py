from sklearn import tree

#data set to train decision tree model on
#list of lists [height, weight, shoe_size] each sublist representing an individual

X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39], [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

#labels of sex for each individual associated to the measurements in X
Y = ['male', 'female', 'female', 'female', 'male', 'male', 'male', 'female', 'male', 'female', 'male']

#variable to store decision tree

clf = tree.DecisionTreeClassifier()
#train on X and Y
clf = clf.fit(X, Y)

prediction = clf.predict([[190, 70, 43]])

print(prediction)