import numpy as np
from abc import ABCMeta, abstractmethod
import scipy.stats as stats
from scipy.special import expit, logit

class AbstractScoreMatrix(object):
    """A rater-by-answer matrix of scores for answers to the question.
    
    Includes two ranking methods:
        (e) anca_rating()
        (j) lizzie_rating()
        
    All of them return a 1D Numpy array with the *scores* of the answers.
    
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
        
        m = len(self.subset["answers"]["data"])
        
        if m < 1:
            raise ValueError("No answers have been rated!")
        
        if len(self.subset["raters"]["data"]) < 1:
            raise ValueError("No users have rated answers!")
            
        if m==1:
            raise ValueError("Only one answer has been rated!")
        
        # remove rows and columns from the rating matrix that contain no data
        score_mat = self.matrix[self.subset['raters']['data'][:,None], 
                                self.subset['answers']['data']]
        
        # initialize matrix of pairwise differences
        kmat = np.zeros((m,m))
        
        # iterate over pairs of answers
        for i in range(m):
            for j in range(i+1, m):

                # find the users who have rated both answers in the pair
                foo = np.all((score_mat[:,[i,j]] > 0), axis=1)

                # count up the number of "wins" and "losses", divide by number 
                # of users who have rated both items. 
                if np.any(foo):
                    sij = np.sum(score_mat[np.where(foo), i] > \
                                 score_mat[np.where(foo), j]) / sum(foo)
                    sji = np.sum(score_mat[np.where(foo), j] > \
                                 score_mat[np.where(foo), i]) / sum(foo)
                    kmat[i,j] = sij - sji
                    kmat[j,i] = sji - sij
        
        # sum up the pairwise differences
        rvec = np.sum(kmat,axis=1)/m
        
        # normalize, so everything that has received a rating will be at least 
        # one point higher than unrated items
        rvec = rvec - np.min(rvec - 1)
        
        # Insert the rated items into a vector containing scores for all items
        all_rvec = np.zeros((len(self.answer_ids)))
        for i in range(len(rvec)):
            all_rvec[self.subset["answers"]["data"][i]] = rvec[i]
            
        return all_rvec
    
    def lizzie_rating(self):
        num_ratings = np.sum(self.matrix >= 0, axis = 0)
        anca_vec = self.anca_rating()
        anca_min = np.min(anca_vec[np.where(anca_vec > 0.0)])
        test_tau, test_p_value = stats.kendalltau(anca_vec, num_ratings)
        
        max_possible_score = len(num_ratings) - 1
        min_possible_score = 1 - len(num_ratings)
        
        if np.isnan(test_tau):
            return anca_vec
        if test_p_value < 0.05 and test_rho > 0:
            # normalize the score vector so that values are between 0 and 1
            svec = anca_vec - np.min(anca_vec)
            max_s = float(np.max(svec))
            if max_s != 0:
                svec = svec / max_s
            
            max_possible_score = (max_possible_score - np.min(anca_vec)) / max_s
            min_possible_score = (min_possible_score - np.min(anca_vec)) / max_s
            
            # calculate number of ratings per answer and normalize
            rvec = num_ratings - np.min(num_ratings)
            rvec = rvec / float(np.max(rvec))

            # sum weighted vecs
            wvec = svec + (rvec * test_rho)
            
            return wvec
        else:
            return anca_vec


class ScoreMatrix(AbstractScoreMatrix):
    """Take an numpy array containing a rater-by-answer matrix of scores, and 
    turn it into a ScoreMatrix object with the two rating aggregation functions.
    Note that -1 indicates missing data.
    """ 
    
    def __init__(self, arr):
        self.matrix = arr
        self.answer_ids = range(self.matrix.shape[1])
        self.rater_ids = range(self.matrix.shape[0])
        self.subset = self.get_missing_subset()
        
        self.anca = self.anca_rating()
        self.lizzie = self.lizzie_rating()

    def get_missing_subset(self):
        """Return a dictionary of the indices of answers that have been scored 
        and answers that haven't (and raters that have provided scores, and 
        raters that havent).
        Used in a preprocessing step in user_preferences() and anca_rating(), 
        to get a subset of the score matrix where unrated items have been 
        dropped (and users who've done no rating have been dropped).
        Retaining a list of indices of the dropped items means we can still 
        create an overall ordering of all items (rated and unrated).
        """
        
        subset = {'answers':{'data':np.where(np.logical_not(
                                    np.all(self.matrix == -1, axis=0)))[0], \
                             'missing':np.where(
                                    np.all(self.matrix == -1, axis=0))[0]}, \
                  'raters':{'data':np.where(np.logical_not(
                                    np.all(self.matrix == -1, axis=1)))[0], \
                            'missing':np.where(
                                    np.all(self.matrix == -1, axis=1))[0]}}
        return subset
    
             
