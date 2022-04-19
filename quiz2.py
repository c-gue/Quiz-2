#github link: https://github.com/c-gue/Quiz-2.git
import pandas as pd
import csv

animal_class = pd.read_csv('animal_classes.csv')
animal_train = pd.read_csv('animals_train.csv')
animal_test = pd.read_csv('animals_test.csv')

x_train = animal_train.iloc[:,:-1]
y_train = animal_train.iloc[:,-1]
x_test = animal_test.iloc[:,1:]
x_animal_name = animal_test.iloc[:,0]

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X=x_train,y=y_train)
predicted = knn.predict(X=x_test)

with open('final_animals.csv',mode='w',newline='') as csv_file:
    header = ['animal_name','prediction']
    writer = csv.DictWriter(csv_file, fieldnames=header, delimiter=',')
    writer.writeheader()

    final_name = []
    final_pred = []
    for i in x_animal_name:
        final_name.append(i)
    for j in predicted:
        if j == animal_class['Class_Number'][j-1]:
            j = animal_class['Class_Type'][j-1]
            final_pred.append(j)

    for final in zip(final_name, final_pred):
        f1,f2 = final
        writer.writerow({'animal_name':f1,'prediction':f2})

print('done')
