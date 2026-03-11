'''
DATA5000
This script is my implementation of Random Forest for gene expression.
'''
# Pandas can be used to read from a csv file, but that can likely just be done in R since I also want to make
# graphs of the data.
from sklearn.ensemble import RandomForestClassifier as RFclassifier
# n_estimators = number of trees in the forest, default 100
# criterion = quality of split
# max_depth = max nodes of a tree
# bootstrap = boolean on whether bootstrapping is used when sampling for trees, default True
# random_state = controls the randomness of bootstrapping and sampling of features to consider when looking for best split
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import classification_report as classify
from sklearn.metrics import confusion_matrix as confmat
from sklearn.metrics import ConfusionMatrixDisplay as confmatDisplay
import matplotlib.pyplot as pyplt
from sklearn.model_selection import train_test_split as tts
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

DEBUG = False
VERBOSE = True

# Read in the gene data
if VERBOSE:
    print("Reading in data now...")
    
data = pd.read_csv("geneData.csv")

if VERBOSE:
    print("Data read successfuly.")

# 19 controls, 190 nonSBI, 89 SBI

# Remove healthy controls from the data since we are not going to be analyzing them
data = data[data.groupData != "condition: Healthy Control"]

# The first 89 nonSBI samples and all SBI samples
dataNonSBI = data[data.groupData == "condition: nonSBI"][:89]
dataSBI = data[data.groupData == "condition: SBI"]
# Concatenate the two data frames into a new data frame.
dataNew = pd.concat([dataNonSBI, dataSBI])

if DEBUG:
    print(len(dataNonSBI))
    print(dataNonSBI)
    print(len(dataSBI))
    print(dataSBI)

    print(len(dataNew))
    print(dataNew)

# 15 genes when comparing against all 3 groups (bacterial, non-bacterial, and control)
X = data[["ILMN_1796316", "ILMN_1762713", "ILMN_1695157", "ILMN_2054019", "ILMN_1714643", "ILMN_1674063", "ILMN_1718558", "ILMN_1707695", "ILMN_1712999", "ILMN_1662358",
          "ILMN_1695404", "ILMN_1654639", "ILMN_1707077", "ILMN_1735058", "ILMN_1810420"]]

# 20, top 10 of each expressing more in one than the other
#            SBI                SBI            SBI              nonSBI        SBI             nonSBI            nonSBI          nonSBI         SBI              nonSBI
X2 = data[["ILMN_1796316", "ILMN_1762713", "ILMN_1695157", "ILMN_2054019", "ILMN_1714643", "ILMN_1674063", "ILMN_1718558", "ILMN_1707695", "ILMN_1712999", "ILMN_1662358",
          #   nonSBI        nonSBI            SBI           nonSBI           SBI              nonSBI         nonSBI             SBI            SBI              SBI
          "ILMN_1695404", "ILMN_1654639", "ILMN_1707077", "ILMN_1735058", "ILMN_1810420", "ILMN_1742618", "ILMN_1653466", "ILMN_1661695", "ILMN_1674394", "ILMN_1704870"]]

# 10 genes when comparing only SBI and nonSBI
X3 = data[["ILMN_2054019", "ILMN_1707695", "ILMN_1674063", "ILMN_1718558", "ILMN_1735058", "ILMN_1662358", "ILMN_1695404", "ILMN_1695157", "ILMN_1654639", "ILMN_1653466"]]

# 10 genes, 5 of each expressing more in one than the other
#                nonSBI           nonSBI          nonSBI         nonSBI           nonSBI          SBI             SBI              SBI              SBI             SBI
X4 = data[["ILMN_2054019", "ILMN_1707695", "ILMN_1674063", "ILMN_1718558", "ILMN_1735058", "ILMN_1662358", "ILMN_1695404"]] #"ILMN_1695157", "ILMN_1714643", "ILMN_1796316", "ILMN_1712999", "ILMN_1674394"]]

Y = data['groupData'] # The parameter we are trying to estimate

if DEBUG:
    print(Y)

# Function to check if input to a variable is valid
def check(var, lower=None, upper=None, valid=None):
    '''
    var - the variable
    lower - a lower bound for the var, if it exists
    upper - an upper bound for the var, if it exists
    valid - a list of valid values for the variable to have, if applicable
    '''

    if DEBUG:
        print(var, lower, upper, valid)
        print(type(var), type(lower), type(upper))

    # Check lower
    if lower is not None:
        if type(var) != type(lower) or var < lower:
            return False
        
    # Check upper
    if upper is not None:
        if type(var) != type(upper) or var > upper:
            return False

    # Check valid
    if valid is not None:
        if var not in valid:
            return False
        
    return True

# Perform random forest
def randomForest(num_runs, test_split, num_trees, state = None):
    '''
    test_split = portion of dataset to be used for testing
    num_trees = number of decision trees in the forest
    state = random state. Used for reproducability. Controls randomness of algorithm. Default None.'''

    # The information to return
    results = [None]*num_runs
    accuracies = [None]*num_runs
    confusion_matrices = [None]*num_runs

    # Loop
    for i in range(num_runs):

        if VERBOSE:
            print("Run " + str(i+1), "started.")

        # random_state controls randomness, used for reproducability
        Xtrain, Xtest, Ytrain, Ytest = tts(X4, Y, test_size=test_split)

        # The fit(X, Y, sample_weight=None) method creates the random forest
        # X are the traning samples and Y are the correct classifications
        rf_classifier = RFclassifier(criterion="entropy", n_estimators=num_trees, max_depth=6, random_state=state)

        if VERBOSE:
            print("Fitting...")

        rf_classifier.fit(Xtrain, Ytrain)

        if VERBOSE:
            print("PRedicting...")

        # Make predictions
        Ypred = rf_classifier.predict(Xtest)

        # Check accuracy of predictions
        accuracies[i] = acc(Ytest, Ypred)
        results[i] = classify(Ytest, Ypred)
        confusion_matrices[i] = confmat(Ytest, Ypred)

        if VERBOSE:
            print("Run #" + str(i+1), "completed.")

    return [results, accuracies, confusion_matrices]


def diagnostics(num_runs, confusion_matrices):
    '''Calculates the average sensitivity (ratio of correct true guesses) and specificity (ratio of correct false guesses).'''

    # Average sensitivity and specificity of all runs
    avgSensitivity = 0
    avgSpecificity = 0

    # GO through all the runs
    for i in range(num_runs):   

        # Record prediction correctness
        trueNeg = confusion_matrices[i][0][0]
        falseNeg = confusion_matrices[i][1][0]
        truePos = confusion_matrices[i][1][1]
        falsePos = confusion_matrices[i][0][1]

        if DEBUG:
            print(confusion_matrices[i])
            print(trueNeg, falseNeg, truePos, falsePos)

        # Calculate sensitivity and specificity
        sensitivity = truePos / (truePos + falseNeg)
        specificity = trueNeg / (trueNeg + falsePos)

        avgSensitivity += sensitivity
        avgSpecificity += specificity

    avgSensitivity = (avgSensitivity / num_runs) * 100
    avgSpecificity = (avgSpecificity / num_runs) * 100

    print("Average sensitivity: " + str(round(avgSensitivity, 2)) + "%")
    print("Average specificity: " + str(round(avgSpecificity, 2)) + "%")


def main():

    runs = input("Number of runs: ")
    runs = int(runs)
    while check(runs, 1) == False:
        print("Invalid input '" + str(runs) + "'. Try again.")
        runs = input("Number of runs: ")
        runs = int(runs)

    split = input("Train / test split (0.1 - 0.99): ")
    split = float(split)
    while check(split, 0.05, 0.99) == False:
        print("Invalid input '" + str(split) + "'. Try again.")
        split = input("Train / test split (0.1 - 0.99): ")
        split = float(split)

    trees = input("Number of trees: ")
    trees = int(trees)
    while check(trees, 1) == False:
        print("Invalid input '" + str(trees) + "'. Try again.")
        trees = input("Number of trees: ")
        trees = int(trees)

    # Perform RF algorithm
    [results, accuracies, confusion_matrices] = randomForest(runs, split, trees)

    if VERBOSE:
        for i in range(runs):
            print("Run #" + str(i+1))
            print(results[i])
            print("Accuracy: " + str(round(accuracies[i]*100, 2)) + "%\n")

    # Run diagnostics
    diagnostics(runs, confusion_matrices)

    # Re-run?
    rerun = input("Run again (Y/N)? ")
    while check(rerun, valid=['Y', 'N']) == False:
        print("Invalid input '" + str(rerun) + "'. Try again.")
        rerun = input("Run again (Y/N)? ")

    while rerun == 'Y':

        runs = input("Number of runs: ")
        runs = int(runs)
        while check(runs, 1) == False:
            print("Invalid input '" + str(runs) + "'. Try again.")
            runs = input("Number of runs: ")
            runs = int(runs)

        split = input("Train / test split (0.1 - 0.99): ")
        split = float(split)
        while check(split, 0.05, 0.99) == False:
            print("Invalid input '" + str(split) + "'. Try again.")
            split = input("Train / test split (0.1 - 0.99): ")
            split = float(split)

        trees = input("Number of trees: ")
        trees = int(trees)
        while check(trees, 1) == False:
            print("Invalid input '" + str(trees) + "'. Try again.")
            trees = input("Number of trees: ")
            trees = int(trees)

        # Perform RF algorithm
        [results, accuracies, confusion_matrices] = randomForest(runs, split, trees)

        if VERBOSE:
            for i in range(runs):
                print("Run #" + str(i+1))
                print(results[i])
                print("Accuracy: " + str(round(accuracies[i]*100, 2)) + "%\n")

        diagnostics(runs, confusion_matrices)

        # Re-run?
        rerun = input("Run again (Y/N)? ")
        while check(rerun, valid=['Y', 'N']) == False:
            print("Invalid input '" + str(rerun) + "'. Try again.")
            rerun = input("Run again (Y/N)? ")


main()
