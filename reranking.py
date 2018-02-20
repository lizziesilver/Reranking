import numpy as np
from abc import ABCMeta, abstractmethod
import scipy.stats as stats
from scipy.special import expit, logit
import sys
import unittest

class AbstractScoreMatrix(object):
    """A rater-by-answer matrix of scores for answers to the question.
    
    Includes two ranking methods:
        (e) anca_rating()
        (j) lizzie_rating()
        
    Both of them return a 1D Numpy array with the *scores* of the answers.
    
    To implement these methods you need a user-by-answer matrix, where each cell  
    contains a single number representing that user's rating of that answer. 
    Ratings are between 0 and 100. 
    
    If the user has not rated an item, I use -1 as the "missing data" value.
    
    This is an abstract class because the instantiation differs depending on  
    where you get the data from (a CSV file, a connection to the SWARM database 
    + a question ID, etc.).
    """ 
    __metaclass__ = ABCMeta
    
    def anca_rating(self):
        """modified version of user preference rankings, suggested by Anca 
        Hanea: just use the win-loss ratio, not the exact point differences.
        """
        
        # start by excluding the answers that have no ratings and the raters
        # that didn't submit any ratings
        subset = {'answers':{'data':np.where(np.logical_not(
                                    np.all(self.matrix == -1, axis=0)))[0], \
                             'missing':np.where(
                                    np.all(self.matrix == -1, axis=0))[0]}, \
                  'raters':{'data':np.where(np.logical_not(
                                    np.all(self.matrix == -1, axis=1)))[0], \
                            'missing':np.where(
                                    np.all(self.matrix == -1, axis=1))[0]}}
        
        m = len(subset["answers"]["data"])
        
        # if no answers have been rated, or no raters have rated anything
        if ((m < 1 or len(subset["raters"]["data"]) < 1) or 
                                 np.all(self.matrix == -1)):
            return np.zeros((self.matrix.shape[1]))
        
        score_mat = self.matrix[subset['raters']['data'][:,None], 
                                subset['answers']['data']]
        
        # initialize matrix of pairwise differences
        kmat = np.zeros((m,m))
        
        # iterate over pairs of answers
        for i in range(m):
            for j in range(i+1, m):

                
                # find the users who have rated both answers in the pair
                both_rated = np.all((score_mat[:,[i,j]] > 0), axis=1)
                
                # count up the number of "wins" and "losses", divide by number 
                # of users who have rated both items. 
                if np.any(both_rated):
                    num_raters = sum(both_rated)
                    both_rated = np.where(both_rated)
                    
                    wins = np.sum(score_mat[both_rated, i] > \
                                  score_mat[both_rated, j]) 
                    losses = np.sum(score_mat[both_rated, j] > \
                                    score_mat[both_rated, i]) 
                    kmat[i,j] = (wins - losses) / num_raters
                    kmat[j,i] = (losses - wins) / num_raters
                
        # sum up the pairwise differences
        rvec = np.sum(kmat,axis=1)/m
        
        # normalize, so everything that has received a rating will be at least 
        # one point higher than unrated items
        rvec = rvec - np.min(rvec - 1)
        
        # Insert the rated items into a vector containing scores for all items
        all_rvec = np.zeros((len(self.answer_ids)))
        for i in range(len(rvec)):
            all_rvec[subset["answers"]["data"][i]] = rvec[i]
            
        return all_rvec
    
    def lizzie_rating(self):
        num_ratings = np.sum(self.matrix >= 0, axis = 0)
        anca_vec = self.anca_rating()
        
        test_tau, test_p_value = stats.kendalltau(anca_vec, num_ratings)
        
        # Conditions to exclude:
        # If either vector is all the same value (e.g. [2,2,2]), test_tau is nan
        # If the test is not statistically significant (because of, e.g., small 
        # number of answers or ratings), we don't want to add a bonus.
        # If the correlation is significant but negative I assume it is a false 
        # positive.
        
        if not np.isnan(test_tau) and (test_p_value < 0.05 and test_tau > 0):
            # normalize the score vector so values are between 0 & 1
            svec = anca_vec - np.min(anca_vec)
            if np.max(svec) != 0:
                svec = svec / float(np.max(svec))
            
            # normalize count of ratings per answer so values are between 0 & 1
            rvec = num_ratings - np.min(num_ratings)
            if np.max(rvec) != 0:
                rvec = rvec / float(np.max(rvec))

            # Add a bonus for number of ratings to the score vector. The size of
            # the bonus is weighted by the correlation between the two vectors.
            wvec = svec + (rvec * test_tau)
            
        else:
            wvec = anca_vec
        
        # now we pin the lowest score to the lowest average rating, and the
        # highest score to the highest average rating, so the numbers look
        # about right to the users:

        # 1. start by calculating the average rating of each response:
        # a. sum of all ratings
        newmat = self.matrix.copy()
        newmat[newmat == -1] = 0
        numerator = np.sum(newmat, axis = 0)
        
        # b. number of ratings received
        denominator = np.sum(self.matrix != -1, axis = 0)
        
        # c. divide to get the average ratings (subset by "hasrating" to avoid div by 0)
        hasrating = np.where(np.logical_not(np.all(self.matrix == -1, axis=0)))[0]
        averages = numerator[hasrating] / denominator[hasrating]
        
        # 2. find the highest and lowest values, if they exist; else set to 0
        if len(averages) > 0:
            low_score = np.min(averages)
            high_score = np.max(averages)
        else:
            low_score = 0
            high_score = 0
        
        # 3. finally, reweight so scores look more intuitive 
        # a. set the min to 0
        wvec = wvec - np.min(wvec)

        # b. set the max to high_score - low_score, if that is not 0
        if (np.max(wvec) != 0) & ((high_score - low_score) != 0):
            wvec = (wvec / np.max(wvec)) * (high_score - low_score)

        # c. shift everything up by low_score
        wvec = wvec + low_score
        
        return wvec


class ScoreMatrix(AbstractScoreMatrix):
    """Take an numpy array containing a rater-by-answer matrix of scores, and 
    turn it into a ScoreMatrix object with the two rating aggregation functions.
    Note that -1 indicates missing data.
    """ 
    
    def __init__(self, arr):
        self.matrix = arr
        self.answer_ids = range(self.matrix.shape[1])
        self.rater_ids = range(self.matrix.shape[0])
        
        self.anca = self.anca_rating()
        self.lizzie = self.lizzie_rating()

################################################################################
# TESTS

def test_init():
    a = np.array([[.7, .57, .4],
                  [.1, .7,  .6],
                  [.2, .5,  .6]])

    test_sm = ScoreMatrix(a)
    assert(np.allclose(test_sm.lizzie, 
        np.array([0.33333333,  0.59,        0.46166667])))

def test_nodata():
    a = np.array([[-1, -1, -1],
                  [-1, -1, -1],
                  [-1, -1, -1]])

    test_sm = ScoreMatrix(a)
    assert(np.array_equal(test_sm.lizzie, 
        np.array([ 0., 0., 0.])))

def test_all_equal():
    a = np.array([[1, 1, 1, 1],
                  [2, 2, 2, 2],
                  [3, 3, 3, 3]])

    test_sm = ScoreMatrix(a)
    val = test_sm.lizzie[0]
    assert(np.all(test_sm.lizzie == val))

def test_chris():
    a = np.array([[35, 10, 32, 68, 50, 68, 19, 56.4, 50., 68],
                  [10, 22,  -1, -1, -1, -1, -1, -1, -1, -1]])

    test_sm = ScoreMatrix(a)
    assert(np.allclose(test_sm.lizzie, 
        np.array([29.86666667, 16., 26.4, 68., 43.73333333, 68.,
                  19.46666667, 54.13333333, 43.73333333, 68.])))


################################################################################
# MAIN
def main(argv):
    test_init()
    test_nodata()
    test_all_equal()
    test_chris()


if __name__ == "__main__":
    main(sys.argv)
