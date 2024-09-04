import pandas as pd
from sklearn.model_selection import train_test_split
from eda import read_csv
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def prepare_data(df, train_size=None, shuffle=True, random_state=None):

    #Αφαιρώ με την μέθοδο drop από το dataset, τις στήλες Month, Browser και Operating Systems.
    df.drop(['Month', 'Browser', 'OperatingSystems'], axis=1, inplace=True)

    # Μετατρέπω τις Boolean τιμές True και False που βρίσκονται στις στήλες Revenue και Weekend αντίστοιχα σε ακεραίους.
    df['Revenue'] = df['Revenue'].astype(int)
    df['Weekend'] = df['Weekend'].astype(int)

    # Εφαρμόζω One-hot encoding στις κατηγορικές μεταβλητές Region, TrafficType και VisitorType καλώντας την μέθοδο get_dummies της βιβλιοθήκης pandas όπως αναφέρεται στην εκφώνηση.
    df = pd.get_dummies(df, columns=['Region', 'TrafficType', 'VisitorType'])

    # Χωρίζω το σύνολο δεδομένων σε X και Y, όπου στο Y θα βρίσκεται η μεταβλητή-στήλη Revenue, που είναι και η μεταβλητή στόχος ενώ στο X θα βρίσκονται όλες οι
    # υπόλοιπες μεταβλητές μου.
    y = df['Revenue']
    X = df.drop('Revenue', axis=1)

    # Χωρίζω το σύνολο δεδομένων μου, σε σύνολο δοκιμής και σύνολο εκπαίδευσης χρησιμοποιώντας όπως αναφέρεται στην εκφώνηση την train_test_split και δίνοντας σαν
    # παραμέτρους αυτές που δόθηκαν για είσοδο στην συνάρτηση prepare_data.
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, shuffle=shuffle,
                                                        random_state=random_state)
    # Επιστρέφω το σύνολο εκπαίδευσης (X_train, y_train) και το σύνολο δοκιμής (X_test, y_test).
    return X_train, X_test, y_train, y_test

filePath = "project2_dataset.csv" # Ορίζω το filepath όπου βρίσκεται το csv αρχείο. Αλάξτε το filepath αναλόγως με την θέση που βρίσκεται το αρχείο στον δικό σας υπολογιστή.
                                  # Εμένα ήταν στον τρέχοντα κατάλογο μαζί με τα αρχεία .py.
csv_data = read_csv(filePath) # Καλώ την συνάρτηση read_csv που βρίσκεται στο αρχείο eda.py και δίνω σαν παράμετρο το filepath του αρχείου δεδομένων παραπάνω,
                              # ώστε η συνάρτηση να μου επιστρέψει τα δεδομένα που διαβάστηκαν.

# Εφόσον διάβασα τα δεδομένα από το αρχείο csv και τα ανέθεσα παραπάνω στην μεταβλητή csv_data, καλώ την prepare_data με 70% των δεδομένων να προορίζονται για
# εκπαίδευση(train_size) και το υπόλοιπο για δοκιμή.
X_train, X_test, y_train, y_test = prepare_data(csv_data, train_size=0.7, shuffle=True, random_state=42)

# Κάνοντας χρήση της MinMaxScaler() κανονικοποιώ τα δεδομένα του συνόλου εκπαίδευσης και δοκιμής για το X.
scaler = MinMaxScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

# Δημιουργώ το μοντέλο μου, δημιουργώντας ένα instance τύπου LogisticRegression. Για τον αριθμό των επαναλήψεων δοκίμαζα κάθε φορά διαφορετικές τιμές,
# ξεκινώντας από το 100 μέχρι να μην έχω σφάλμα. Συγκεκριμένα δοκίμασα το 100, το 200, το 250, το 300, το 350 και το 400. Όταν είδα
# ότι το μοντέλο συγκλίνει στις 400 επαναλήψεις, δοκίμασα να δω ποιος είναι ο 'πρώτος' αριθμός επαναλήψεων που συγκλίνει και κατεβαίνοντας ανά 5 έφτασα στις 365. Ενώ,
# στις 370 σύγκλινε, ύστερα στις 365 δεν σύγκλινε, οπότε δοκίμασα τιμές από το 365 έως και το 370 και βρήκα ότι ο 'πρώτος' αριθμός επαναλήψεων που συγκλίνει είναι
# οι 367. Στο instance-αντικείμενο που δημιουργώ, θέτω όπως ορίζεται στην εκφώνηση την παράμετρο penalty σε None.
log_reg = LogisticRegression(solver='lbfgs',penalty=None,max_iter=367)
log_reg.fit(X_train_normalized, y_train) # Εκπαιδεύω το μοντέλο μου καλώντας την μέθοδο fit, πάνω στα δεδομένα (σύνολο) εκπαίδευσης.

y_pred_train = log_reg.predict(X_train_normalized) # Καλώντας την predict πάνω στο κανονικοποιημένο σύνολο εκπαίδευσης, καταγράφω τις προβλέψεις του, y_pred_train.
y_pred_test = log_reg.predict(X_test_normalized) # Καλώντας την predict πάνω στο κανονικοποιημένο σύνολο δοκιμής, καταγράφω τις προβλέψεις του, y_pred_test.

train_accuracy = accuracy_score(y_train, y_pred_train) # Προκειμένου να υπολογίσω την ευστοχία του μοντέλου στο σύνολο εκπαίδευσης καλώ την accuracy_score και δίνω
                                                       # σαν παράμετρο τις προβλέψεις του μοντέλου αλλά και τα κανονικά δεδομένα. Η μέθοδος αυτή θα μου επιστρέψει
                                                       # έναν αριθμό από το 0 έως το 1, ο οποίος θα δείχνει πόσο θα συγκλίνουν οι προβλέψεις του μοντέλου σε σχέση με τα πραγματικά
                                                       # δεδομένα. Όσο μεγαλύτερος είναι αυτός ο αριθμός τόσο μεγαλύτερη είναι και η ευστοχία του μοντέλου.
test_accuracy = accuracy_score(y_test, y_pred_test)    # Παρόμοια διαδικασία ακολουθώ για να βρω την ευστοχία του μοντέλου και στο σύνολο δοκιμής.


# Εκτύπωση αποτελέσματος για την ερώτηση 5.1. Ευστοχία μοντέλου στο σύνολο εκπαίδευσης.
print(f"Question 5.1: Accuracy on the training set: {np.round(train_accuracy,2)} or {np.round(train_accuracy * 100, 2)}%")
# Εκτύπωση αποτελέσματος για την ερώτηση 5.2. Ευστοχία μοντέλου στο σύνολο δοκιμής.
print(f"Question 5.2: Accuracy on the test set: {np.round(test_accuracy,2)} or {np.round(test_accuracy * 100, 2)}%")

# Δημιουργώ τον πίνακα σύγχυσης καλώντας την μέθοδο confusion_matrix.
conf_matrix = confusion_matrix(y_test, y_pred_test)
# Εκτύπωση αποτελέσματος για την ερώτηση 5.3. Πίνακας σύγχυσης.
print("Question 5.3: Confusion Matrix:")
print(conf_matrix) # Εκτυπώνω τον πίνακα.
# Εμφάνιση σε μορφή διαγράμματος του πίνακα σύγχυσης.
conf_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
conf_display.plot()
plt.show()

# precision = precision_score(y_test, y_pred_test)
# recall = recall_score(y_test, y_pred_test)
# f1 = f1_score(y_test, y_pred_test)

# print(f"Precision on the test set: {precision:.2f}")
# print(f"Recall on the test set: {recall:.2f}")
# print(f"F1-Score on the test set: {f1:.2f}")