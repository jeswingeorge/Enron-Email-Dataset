#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where
        each tuple is of the form (age, net_worth, error).
    """
    import numpy as np
    cleaned_data = []

    # your code goes here
    #errors = (predictions - net_worths)**2
    errors = np.square(predictions-net_worths)
    max_error = np.percentile(errors, 90)
    #t = errors.copy()
    #top_error = np.array( [] )
    #c = 0

    #while (c < length):
    #    ind = t.argmax()
    #    print t[ind]
    #    top_error = np.insert(top_error, c, t[ind])
    #    t = np.delete(t, ind)
    #    c += 1

    for i in range(len(errors)):
        if errors[i] < max_error:
            a = (ages[i], net_worths[i], errors[i])
            cleaned_data.append(a)

    return cleaned_data
