import sys
import getopt
import CR_CrossValidation

if __name__ == '__main__':

    alpha = 0.5
    c = 0.85
    max_iter = 1000
    epsilon = 1e-6
    dataset = "../data/P_G_NoN.npy"

    opts, args = getopt.getopt(sys.argv[1:], "h", ["alpha=", "c=", "max_iter=", "epsilon=", "dataset="])
    for option, value in opts:
        if option == "-h":
            print "Welcome, this is a program of NoN Cross Ranking"
            print ""
            print "python __init__.py [option] [argument]"
            print ""
            print "Option and arguments:"
            print "--alpha          The regularization parameter for cross-network consistency."
            print "--c              The regularization parameter for query preference."
            print "--max_iter       The maximal number of iteration for updating ranking vector."
            print "--epsilon        The convergence parameter."
            print "--dataset        The path for the dataset."
            print ""
            print "Example:"
            print "python __init__.py --alpha 0.5 --c 0.85 --max_iter 1000 --epsilon 1e-6 --dataset ../data/P_G_NoN.npy"
            print ""
            exit(0)
        if option == "--alpha":
            alpha = float(value)
        if option == "--c":
            c = float(value)
        if option == "--max_iter":
            max_iter = int(value)
        if option == "--epsilon":
            epsilon = float(value)
        if option == "--dataset":
            dataset = value

    CR_CrossValidation.cr_cross_validation(alpha, c, max_iter, epsilon, dataset)
