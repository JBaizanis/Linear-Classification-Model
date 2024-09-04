import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Η συνάρτηση read_csv παίρνει ως παράμετρο ένα filepath για την τοποθεσία του αρχείου project2_dataset, όπου περιέχει το σύνολο των δεδομένων που θέλουμε να διαβάσουμε.
def read_csv(filePath):

    dataframe = pd.read_csv(filePath) # Με την read_csv διαβάζω τα περιεχόμενα του αρχείου project2_dataset.
    records = len(dataframe)
    print("Question 1.1: Number of records of the dataset: ", records) # Εκτύπωση αποτελέσματος για την ερώτηση την 1.1. Συνολικός αριθμός εγγραφών.
    result = dataframe['Revenue'].value_counts()[True]  # Για την στήλη Revenue η οποία αναπαριστά αν η επίσκεψη στον διαδικτυακό ιστότοπο
                                                        # κατέληξε σε αγορά ή όχι, βλέπω αν η τιμή της είναι True, δηλαδή αν η επίσκεψη κατέληξε σε αγορά.
                                                        # Σε αυτήν την περίπτωση μετρώ το σύνολο των εγγραφών (records) αυτών με την value_counts().
    buy_percentage=(result/records)*100
    # Εκτύπωση αποτελέσματος για την ερώτηση 1.2. Ποσοστό χρηστών που τελικά αγόρασαν με την επίσκεψη τους στο ηλεκτρονικό κατάστημα.
    print(f"Question 1.2: Percentage of visitors who purchased something during their visit, among the visitors: {np.round(buy_percentage, 2)}%")

    false_count = dataframe['Revenue'].value_counts()[False]
    accuracy = false_count/records # Για να υπολογίσω την ευστοχία, μετρώ τα False που υπάρχουν ανάμεσα σε όλες τις εγγραφές.
    # Εκτύπωση αποτελέσματος για την ερώτηση 1.3. Ευστοχία μοντέλου το οποίο προβλέπει πάντα ότι ο χρήστης δεν θα αγοράσει.
    print(f"Question 1.3: Accuracy of the model which always predicts revenue as False is: {np.round(accuracy, 2)} or {np.round(accuracy * 100, 2)}%")

    # Εντοπίζω τις αριθμητικές και κατηγορικές μεταβλητές - στήλες από το σύνολο δεδομένων.
    numerical_columns = dataframe.select_dtypes(include=['number']).columns
    categorical_columns = dataframe.select_dtypes(exclude=['number']).columns

    # Δημιουργία φακέλου για την αποθήκευση των γραφημάτων παρακάτω.
    if not os.path.exists('charts'):
        os.makedirs('charts')

    # Τυπώνω χρησιμοποιώντας την βιβλιοθήκη matplotlib.pyplot για κάθε αριθμητική μεταβλητή ένα ιστόγραμμα, το οποίο θα δείχνει για κάθε τιμή της αριθμητικής
    # μεταβλητής ποια είναι η συχνότητα αυτής της τιμής.
    for column in numerical_columns:
        plt.figure(figsize=(6, 4))
        dataframe[column].hist(bins=30)
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.savefig(f'charts/{column}_distribution.png')
        plt.close()

    # Τυπώνω χρησιμοποιώντας την βιβλιοθήκη matplotlib.pyplot για κάθε κατηγορική μεταβλητή μπάρες, οι οποίες π.χ για την μεταβλητή Revenue που έχω True και False τιμές,
    # θα δείχνουν πόσα είναι τα True και πόσα τα False. Πόσες δηλαδή επισκέψεις στο διαδικτυακό ιστότοπο κατέληξαν σε αγορά και πόσες όχι. Αντίστοιχα και για τις υπόλοιπες κατηγορικές
    # μεταβλητές.
    for column in categorical_columns:
        plt.figure(figsize=(6, 4))
        dataframe[column].value_counts().plot(kind='bar')
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.savefig(f'charts/{column}_distribution.png')
        plt.close()

    # Διαγράμματα που δείχνουν την σχέση μεταξύ των αριθμητικών μεταβλητών και της μεταβλητής-στήλης Revenue που είναι η μεταβλητή στόχος.
    for column in numerical_columns:
        if column != 'Revenue':
            plt.figure(figsize=(6, 4))
            dataframe.boxplot(column=column, by='Revenue')
            plt.title(f'{column} by Revenue')
            plt.xlabel('Revenue')
            plt.ylabel(column)
            plt.suptitle('')
            plt.savefig(f'charts/{column}_by_Revenue.png')
            plt.close()

    # Διαγράμματα που δείχνουν την σχέση μεταξύ των κατηγορικών μεταβλητών και της μεταβλητής-στήλης Revenue που είναι η μεταβλητή στόχος.
    for column in categorical_columns:
        if column != 'Revenue':
            plt.figure(figsize=(6, 4))
            pd.crosstab(dataframe[column], dataframe['Revenue']).plot(kind='bar', stacked=True)
            plt.title(f'{column} by Revenue')
            plt.xlabel(column)
            plt.ylabel('Count')
            plt.savefig(f'charts/{column}_by_Revenue.png')
            plt.close()

    print("Charts have been saved in the current directory in charts folder.")
    return dataframe