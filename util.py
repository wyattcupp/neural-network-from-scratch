'''
Wyatt Cupp
wyattcupp@gmail.com


This file contains utility functions and helper methods.
'''

def batch_iterator(X, y, batch_size):
        '''
        Represents an iterator for a batch of samples using a generator.
        '''
        i = 0
        while i < len(X):
            if i + batch_size >= len(X):
                yield X[i:], y[i:]
            else:
                yield X[i:i+batch_size], y[i:i+batch_size]

            i += batch_size



