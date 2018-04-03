import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt

def split_dataset(dataset, train_percentage, feature_headers, target_header):
    """
    Split the dataset with train_percentage
    :param dataset:
    :param train_percentage:
    :param feature_headers:
    :param target_header:
    :return: train_x, test_x, train_y, test_y
    """

    # Split dataset into train and test dataset
    train_x, test_x, train_y, test_y = train_test_split(dataset[feature_headers], dataset[target_header],
                                                        train_size=train_percentage)
    return train_x, test_x, train_y, test_y

def random_forest_classifier(features, target):
    """
    To train the random forest classifier with features and target data
    :param features:
    :param target:
    :return: trained random forest classifier
    """
    clf = RandomForestClassifier()
    clf.fit(features, target)
    return clf


def dataset_statistics(dataset):
    """
    Basic statistics of the dataset
    :param dataset: Pandas dataframe
    :return: None, print the basic statistics of the dataset
    """
    print (dataset.describe())








if __name__=='__main__':
    
    shp=(1177, 1017, 10)
    
    headers = ['$I_{hh}$','$I_{hv}$','$I_{vv}$','$\lambda_{1}$','$\lambda_{2}$','$\lambda_{3}$','PD', '$R_{CO}X$','$I_{CO}X$','det(C3)']

    image_csv_dir='/home/anurag/Documents/MScProject/Meetings_ITC/Results/Classification/image_df_Win_9_corr_False.csv'
    image_df=pd.DataFrame.from_csv(image_csv_dir, sep=',')


    class_leg={'Oil':1,'Water':2}
    TR_csv_dir = '/home/anurag/Documents/MScProject/Meetings_ITC/Results/Classification/TR_Win_9_corr_False.csv'

    TR = pd.DataFrame.from_csv(TR_csv_dir, sep=',')
    
    train_x, test_x, train_y, test_y = split_dataset(TR, 0.7, headers, 'class_id')
    
    print ("Train_x Shape :: ", train_x.shape)
    print ("Train_y Shape :: ", train_y.shape)
    print ("Test_x Shape :: ", test_x.shape)
    print ("Test_y Shape :: ", test_y.shape)
    
    trained_model = random_forest_classifier(train_x, train_y)
    
    predictions = trained_model.predict(test_x)
    #predictions = trained_model.predict(image_df)
    
    #plt.imshow(predictions.reshape(shp[0], shp[1]))
    #plt.show()
    
    for i in range(0, 5):
        print ("Actual outcome :: {} and Predicted outcome :: {}".format(list(test_y)[i], predictions[i]))
    
    print ("Train Accuracy :: ", accuracy_score(train_y, trained_model.predict(train_x)))
    print ("Test Accuracy  :: ", accuracy_score(test_y, predictions))
    print (" Confusion matrix ", confusion_matrix(test_y, predictions))
    print(cohen_kappa_score(test_y, predictions))
