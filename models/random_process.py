import numpy as np
from scipy.stats import multivariate_normal as mv
from scipy.optimize import minimize, dual_annealing


# def rbf_kernel(x1, x2, sigma, width): 
#     n = len(x1)
#     m = len(x2)
#     var = np.array([])
    
#     for i in x1:
#         exponent = np.linalg.norm((x2-i)/width, axis=1)**2
#         var = np.concatenate((var,sigma*np.exp(-0.5*exponent)))
    
#     return var.reshape(n,m)



def rbf_kernel(x1, x2, eta, width):
    tmp_sq = np.square(x1)[:, np.newaxis] + np.square(x2)[np.newaxis] - 2 * np.outer(x1, x2)
    cov = eta * np.exp(-0.5 * tmp_sq / width / width)
    
    return cov



class GPRegression:
    def __init__(self, init_x, init_y, min_x, max_x, err=1e-6):
        self.x = init_x
        self.y = init_y
        
        self.min = min_x
        self.max = max_x
        
        self.err = err
    
    
    def optimize(self, y):
        # res = dual_annealing(self.neg_log_likelihood(y), bounds=((1e-2, 10), (1e-2, 10)),
        #                      maxiter=10) 

        res = minimize(self.neg_log_likelihood(y), [1, 1], bounds=((1e-3, 10), (1e-3, 10)), method='L-BFGS-B')

        params = res.x
        
        return params
    
    
    def neg_log_likelihood(self, y):

        def nll(params):
            Ke = self.cov(params)
            Ke_inv = np.linalg.inv(Ke)
            Ke_det = np.linalg.det(Ke)
            
            ll = -0.5*y.T@Ke_inv@y-np.log(Ke_det)-0.5*len(y)*np.log(2*np.pi)
        
            return -ll

        return nll
        
    
    def cov(self, params):
        sigma = params[0]
        length = params[:1]
        N = len(self.x)
        
        K = rbf_kernel(self.x, self.x, sigma, length)
        
        return K + self.err*np.eye(N)
    
    
    def sample(self, n):
        t = np.arange(self.min, self.max+(self.max-self.min)/n, (self.max-self.min)/n)
        
        return t
    
    
    def get_traj(self, n_test, n_traj=5, err=1e-8):
        dim = len(self.y[0])

        est = []        
        for j in range(n_traj):
            traj = []
            for i in range(dim):
                y = self.y[:, i]
                # print(y)
                # params = self.optimize(y)
                params = np.array([0.0002, 2.2]) # trade off: optimize value, traj strictly pas through the points but bit messy (high freq>> small lengh scale)
                print(params)
                
                sigma = params[0]
                length = params[1:]
                
                Ke = self.cov(params)
                Ke_inv = np.linalg.inv(Ke)
                test_x = self.sample(n_test)
                Ks = rbf_kernel(test_x, self.x, sigma, length)
                
                mean_post = Ks@Ke_inv@y
                cov_post = rbf_kernel(test_x, test_x, sigma, length)-Ks@Ke_inv@Ks.T
                jitter = err*np.eye(len(cov_post))
                # print(np.all(np.linalg.eigvals(cov_post) > 0))
                damper = 1

                sample = mv(mean_post, damper*(cov_post+jitter)).rvs(1)

                # if i == dim-1:
                #     print(mean_post + 1.96*np.diag(cov_post))
                #     print(mean_post - 1.96* np.diag(cov_post))
                #     print(np.diag(cov_post))
                    # print(np.diag(cov_post))
                    # count = 0
                    # while True:
                    #     count += 1
                    #     # print(count)
                    #     const = np.all(sample >= 0.025)
                    #     if const:
                    #         break

                traj.append(sample)

            est.append(np.array(traj))
            
        return est