import collections

############################################################
# Problem 3a

def computeMaxWordLength(text):
    """
    Given a string |text|, return the longest word in |text|.  If there are
    ties, choose the word that comes latest in the alphabet. There won't be 
    puctuations and there will only be splits on spaces. You might find
    max() and list comprehensions handy here.
    """
    # BEGIN_YOUR_CODE (around 1 line of code expected)
    raise Exception("Not implemented yet")
    # END_YOUR_CODE

############################################################
# Problem 3b

def createExistsFunction(text):
    """
    Given a text, return a function f, where f(word) returns whether |word|
    occurs in |text| or not.  f should run in O(1) time.  You might find it
    useful to use set().
    """
    # BEGIN_YOUR_CODE (around 4 lines of code expected)
    raise Exception("Not implemented yet")
    # END_YOUR_CODE

############################################################
# Problem 3c

def manhattanDistance(loc1, loc2):
    """
    Return the Manhattan distance between two locations, where locations are
    pairs (e.g., (3, 5)).
    """
    # BEGIN_YOUR_CODE (around 1 line of code expected)
    raise Exception("Not implemented yet")
    # END_YOUR_CODE

############################################################
# Problem 3d

def sparseVectorDotProduct(v1, v2):
    """
    Given two sparse vectors |v1| and |v2|, each represented as Counters, return
    their dot product.
    You might find it useful to use sum() and a list comprehension.
    """
    # BEGIN_YOUR_CODE (around 4 lines of code expected)
    raise Exception("Not implemented yet")
    # END_YOUR_CODE

############################################################
# Problem 3e

def incrementSparseVector(v1, scale, v2):
    """
    Given two sparse vectors |v1| and |v2|, perform v1 += scale * v2.
    """
    # BEGIN_YOUR_CODE (around 2 lines of code expected)
    raise Exception("Not implemented yet")
    # END_YOUR_CODE

############################################################
# Problem 3f

def computeMostFrequentWord(text):
    """
    Splits the string |text| by whitespace and returns two things as a pair: 
        the set of words that occur the maximum number of times, and
	their count, i.e.
	(set of words that occur the most number of times, that maximum number/count)
    You might find it useful to use collections.Counter().
    """
    # BEGIN_YOUR_CODE (around 5 lines of code expected)
    raise Exception("Not implemented yet")
    # END_YOUR_CODE

############################################################
# Problem 3g

def computeLongestPalindrome(text):
    """
    A palindrome is a string that is equal to its reverse (e.g., 'ana').
    Compute the length of the longest palindrome that can be obtained by deleting
    letters from |text|.
    For example: the longest palindrome in 'animal' is 'ama'.
    Your algorithm should run in O(len(text)^2) time.
    You should first define a recurrence before you start coding.
    """
    # BEGIN_YOUR_CODE (around 19 lines of code expected)
    raise Exception("Not implemented yet")
    # END_YOUR_CODE
