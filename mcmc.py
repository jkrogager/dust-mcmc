import numpy as np

def save_pars(fname, par_dict, var_parnames):
    """Save parameter dictionary to file"""
    with open(fname, 'w') as output:
        for parname in var_parnames:
            val = par_dict[parname]
            if val is None:
                output.write(" %s : None \n" % (parname))
            else:
                output.write(" %s : %.3e \n" % (parname, val))
        output.write("\n# -- Fixed parameters below --\n")
        for key, val in par_dict.items():
            if key not in var_parnames:
                if val is None:
                    output.write(" %s : None \n" % (key))
                else:
                    output.write(" %s : %.3e \n" % (key, val))


class GaussianPrior():
    def __init__(self, mu, sig):
        self.mu = mu
        self.sig = sig
        self.descr = 'Gaussian(%r, %r)' % (mu, sig)
        self.name = 'Gaussian'

    def __call__(self, x):
        # Return the logarithm of the prior probability
        return -0.5*((x - self.mu)/self.sig)**2


class Parameter():
    def __init__(self, value, limits=(-np.inf, np.inf), variable=True,
                 prior=None, label='', tie=None):
        self.value = value
        self.limits = limits
        self.variable = variable
        self.prior = prior
        self.label = label
        self.tie = tie

    def evaluate_prior(self, x):
        if callable(self.prior):
            return self.prior(x)
        else:
            return 0.

    def __str__(self):
        return f"<Parameter {self.label} = {self.value:.3f}, variable: {self.variable}>"


def log_prior(theta, var_names, all_pars):
    lp = 0.
    for parval, parname in zip(theta, var_names):
        par_min, par_max = all_pars[parname].limits
        if par_min <= parval <= par_max:
            lp += all_pars[parname].evaluate_prior(parval)
        else:
            return -np.inf
    return lp

