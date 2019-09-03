import numpy as np

# Get the boosted training data for t passes
def boost(t):
    print("Performing boosting... ")
    c = [] # Our classifier to be returned
    # Read training data as a 2D array
    f = open('pa4train.txt', 'r')
    train = [[int(i) for i in line.strip().split()] for line in f]
    # Initialize distribution over training data, 1/n for all i = 1, 2, ..., n
    d = [1 / len(train)] * len(train)
    # Iterate our boosting algorithm t times
    for i in range(t):
        # Get errs as a 3-tuple of error, word, and label
        errs = [(_err(train, _h(train, i, label), d), i, label) for i in range(4003) for label in [1,-1]]
        err, word, label = min(errs)
        # Continue if a weak learner
        if (err >= 0.5):
            break
        # Calculate alpha
        alpha = 1 / 2 * np.log((1 - err) / err)
        # Calculate h_t
        h = _h(train, word, label)
        # Update distribution
        d = [d[i] * np.exp(-alpha * train[i][-1]* h[i]) for i in range(len(d))]
        z = sum(d)
        d = [d[i] / z for i in range(len(d))]
        # Add to our classifier
        c += [(alpha, word, label)]
    return c

# Calculate rule of weak learner
def _h(training_data, word, label):
    list1 = [label * (1 if line[word] else -1) for line in training_data]
    return list1

# Calculate given error
def _err(training_data, h, d):
    err = np.dot(d, [h[i] != training_data[i][-1] for i in range(len(h))])
    return err

# Find training, validation, or test error
def get_error(c, filename):
    mistakes = 0
    # Read file as a 2D array
    f = open(filename, 'r')
    data = [[int(i) for i in line.strip().split()] for line in f]
    for line in data:
        sum = 0
        for (alpha, word, sign) in c:
            h = 1 * sign if line[word] else -1 * sign
            sum += alpha * h
        if sum * line[-1] < 0:
            mistakes += 1
    return float(mistakes) / len(data)

# Find error
if __name__ == '__main__':
    print("Getting training and testing error... ")
    for i in [3, 7, 10, 15, 20]:
        c = boost(i)
        print("t: ", i)
        print("Training error: ", get_error(c, 'pa4train.txt'))
        print("Testing error: ", get_error(c, 'pa4test.txt'))
    print("Finding first 10 words of the weak learners... ")
    f = open("pa4dictionary.txt", "r")
    dic = [word.strip() for word in f]
    c10 = boost(10)
    results = [dic[word] for (err, word, label) in c10]
    print(results)
