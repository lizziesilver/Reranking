import numpy as np
from abc import ABCMeta, abstractmethod
import scipy.stats as stats
from scipy.special import expit, logit
import sys

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
        if (m < 1 or len(subset["raters"]["data"]) < 1) or np.all(self.matrix == -1):
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
                    num_pairs = sum(both_rated)
                    both_rated = np.where(both_rated)
                    
                    wins = np.sum(score_mat[both_rated, i] > score_mat[both_rated, j]) 
                    losses = np.sum(score_mat[both_rated, j] > score_mat[both_rated, i]) 
                    kmat[i,j] = (wins - losses) / num_pairs
                    kmat[j,i] = (losses - wins) / num_pairs
                
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
        # If either vector is all the same value (e.g. [2,2,2]), the test returns nan.
        # If the test is not statistically significant (because of, e.g., small number 
        # of answers or ratings), we don't want to add a bonus.
        # If the correlation is significant but negative I assume it is a false positive.
        
        if not np.isnan(test_tau) and (test_p_value < 0.05 and test_tau > 0):
            # start by normalizing the score vector so that values are between 0 and 1
            svec = anca_vec - np.min(anca_vec)
            svec = svec / float(np.max(svec))
            
            # normalize the count of ratings per answer so values are between 0 and 1
            rvec = num_ratings - np.min(num_ratings)
            rvec = rvec / float(np.max(rvec))

            # Add a bonus for number of ratings to the score vector. The size 
            # of the bonus is weighted by the correlation between the two vectors.
            wvec = svec + (rvec * test_tau)
            
        else:
            wvec = anca_vec
        
        # reweight so scores look more intuitive 
        # (note that this doesn't change the ordering!)
        wvec = wvec - np.min(wvec) + 0.15 * np.mean(wvec)
        wvec = (wvec / (np.max(wvec) + 0.15 * np.mean(wvec)))
        
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
        np.array([ 0.26623639,  0.91165793,  0.58894716])))


################################################################################
# MAIN
def main(argv):
    test_init()

if __name__ == "__main__":
    main(sys.argv)
