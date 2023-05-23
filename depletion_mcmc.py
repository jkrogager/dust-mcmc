"""
MCMC sampling to recover metallicity for each component of an absorption system
if the overall depletion strength for each component is constrained.

For more information about the parameters, run:
    python depletion_mcmc.py -h

"""

__author__ = 'Jens-Kristian Krogager'
__email__ = 'jens-kristian.krogager@univ-lyon1.fr'

from argparse import ArgumentParser
import emcee
import corner
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from VoigtFit.utils import depletion, Asplund
from scipy.optimize import curve_fit
import os

from report import Report
from mcmc import Parameter, GaussianPrior, log_prior, save_pars


def line(x, a, b):
    return a + b*x


def run_mcmc_sampler(filename, steps=5000, burn_in=500, nwalkers=100, H_WEIGHT=100, output_dir='output',
                     fix_depletion=False):

    bfname = os.path.basename(filename)
    basename = os.path.splitext(bfname)[0]
    report = Report()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read the input parameters from the file:
    _float = np.vectorize(float)
    with open(filename) as file:
        lines = [line.rstrip() for line in file if line[0]=='#']
    input_pars = [l.strip('# ').split(', ') for l in lines if l[-1].isnumeric()]
    if len(input_pars) > 0:
        input_pars = _float(input_pars)
    else:
        input_pars = []

    # Load dataset:
    data = Table.read(filename, comment='#')
    X = data['ion']
    logNHI = data['total'][X == 'H']
    logNHI_err = data['total_err'][X == 'H']
    logN_tot = data['total'][X != 'H']
    logN_err = data['total_err'][X != 'H']
    X = X[X != 'H']

    solar = np.array([Asplund.solar[ion][0] for ion in X])
    # solar_err = np.array([Asplund.solar[ion][1] for ion in X])
    B2 = np.array([depletion.B2[ion] for ion in X])
    N_comps = len([par for par in data.colnames if 'comp' in par]) // 2

    if 'Zn' in X:
        logZ_global = float(logN_tot[X == 'Zn'] - logNHI - (solar[X == 'Zn'] - 12))
    elif 'S' in X:
        logZ_global = float(logN_tot[X == 'S'] - logNHI - (solar[X == 'S'] - 12))
    elif 'Si' in X:
        logZ_global = float(logN_tot[X == 'Si'] - logNHI - (solar[X == 'Si'] - 12))
    else:
        logZ_global = float(logN_tot[X == 'Fe'] - logNHI - (solar[X == 'Fe'] - 12))

    # Determine the slope of the depletion sequence for each component
    all_delta = []
    all_delta_err = []
    all_logN = []
    all_errors = []
    all_pars = []
    for num in range(1, N_comps+1):
        logn = data['comp%i' % num][1:]
        err = data['comp%i_err' % num][1:]
        if hasattr(logn, 'mask'):
            mask = ~logn.mask
            all_logN.append(logn.filled(np.nan))
            all_errors.append(err.filled(np.nan))
        else:
            mask = np.ones_like(logn, dtype=bool)
            all_logN.append(logn)
            all_errors.append(err)
        y = logn - solar + 12
        p_opt, pcov = curve_fit(line, B2[mask], y[mask], sigma=err[mask])
        all_pars.append(p_opt)
        all_delta.append(p_opt[1])
        all_delta_err.append(np.sqrt(pcov[1, 1]))

    all_delta = np.array(all_delta)
    all_delta_err = np.array(all_delta_err)


    # Plotting
    B_plotting = np.linspace(-1.8, 0., 100)

    plt.close('all')
    colors = [plt.colormaps.get('rainbow')(int(x)) for x in np.linspace(0, 255, len(all_delta))]

    fig_depl = plt.figure()
    ax = fig_depl.add_subplot(111)
    ax.plot(B2, logN_tot - solar + 12, 'k+', label='Total')
    report.add_line("Best-fit Depletions:")
    for num, logN_comp in enumerate(all_logN):
        delta_label = '$\delta=%.2f$' % all_delta[num]
        if len(input_pars) > 0:
            delta_label += ' (in: %.2f)' % (input_pars[num, 2])
        ax.errorbar(B2, logN_comp - solar + 12, all_errors[num],
                    ls='', alpha=0.5, color=colors[num])
        ax.plot(B_plotting, line(B_plotting, *all_pars[num]), ls='--', color=colors[num],
                alpha=0.5, label=delta_label)
        report.add_line("  comp%i : %.3f ± %.3f" % (num+1, all_delta[num], all_delta_err[num]))
        print("  comp%i : %.3f ± %.3f" % (num+1, all_delta[num], all_delta_err[num]))
    ax.legend()
    ax.set_xlabel("B2$_X$")
    ax.set_ylabel("logN(X) - X$_{\odot}$ + 12")
    plt.tight_layout()
    fig_depl.savefig(f"{output_dir}/{basename}_depletion.pdf")
    if fix_depletion:
        report.add_line("Using fixed depletions")
        print("Using fixed depletions")
    else:
        report.add_line("Fitting depletions using priors")
        print("Fitting depletions using priors")


    # Define the model based on the total H column:
    def model(B, X_sun, delta, logZ, f_gas):
        # Recast vectors to matrices to vectorize the calculation:
        ones = np.ones(len(B))
        logZ = np.outer(logZ, ones)
        delta = np.outer(delta, ones)
        f_gas = np.outer(f_gas, ones)
        logN_i = logZ + X_sun - 12 + logNHI + np.log10(f_gas) + B*delta
        return np.log10(np.sum(10**logN_i, axis=0))


    def log_likelihood(theta, all_pars):
        if fix_depletion:
            logZ = theta
            delta = np.array([par.value for parname, par in all_pars.items() if 'delta' in parname])
        else:
            nhalf = len(theta) // 2
            logZ = theta[:nhalf]
            delta = theta[nhalf:]
        f_gas = np.random.uniform(1, H_WEIGHT, len(logZ))
        f_gas = f_gas / np.sum(f_gas)
        m = model(B2, solar, delta, logZ, f_gas)
        return -0.5*np.sum(((logN_tot - m)/logN_err)**2)
        

    def log_probability(theta, var_names, all_pars):
        """
        theta : np.array[float]
            an array of parameter values

        par_names : np.array[str]
            an array of parameter names from parameter dictionary

        all_pars : dict[str: Parameter]
            full parameter dictionary with each entry being an instance
            of the Parameter class
        """
        lp = log_prior(theta, var_names, all_pars)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(theta, all_pars)


    # Define parameters:
    parameters = {}
    for num in range(1, N_comps+1):
        parname = 'logZ_%i' % num
        par = Parameter(value=logZ_global,
                        limits=(-5., 2.),
                        label="[M/H]$_{%i}$" % num,
                        )
        parameters[parname] = par

    for num in range(1, N_comps+1):
        parname = 'delta_%i' % num
        par = Parameter(value=all_delta[num-1],
                        limits=(0., 10),
                        label="${\delta}_{%i}$" % num,
                        prior=GaussianPrior(all_delta[num-1], all_delta_err[num-1]),
                        variable=not fix_depletion,
                        )
        parameters[parname] = par

    # -- Get active parameters:
    #    initial parameter values
    theta_0 = np.array([par.value for par in parameters.values() if par.variable])
    #    variable parameter names
    variable_parnames = np.array([parname for parname, par in parameters.items() if par.variable])
    #    variable parameter labels
    active_labels = [par.label for par in parameters.values() if par.variable]


    # -- Set up the dimensions and the walkers:
    assert len(theta_0) == len(variable_parnames), "Wrong number of parameters!"
    ndim = len(theta_0)
    pos = theta_0 + 1.e-2 * np.random.randn(nwalkers, ndim)

    # -- Initiate the sampler:
    print(" Initiating MCMC sampler:")
    print("  Number of dimensions:  %i" % ndim)
    print("  Number of walkers:  %i" % nwalkers)
    print("  Number of steps  :  %i" % steps)
    print(" - Running MCMC chain...")
    report.add_line(" MCMC parameters:\n")
    report.add_line("   N_walkers = %i\n" % nwalkers)
    report.add_line("   steps = %i\n" % steps)
    report.add_line("   burn-in = %i\n" % burn_in)
    report.add_line("\n")
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                    args=[variable_parnames, parameters])
    sampler.run_mcmc(pos, steps, progress=True)
    print("  Discarding first %i steps as burn-in" % burn_in)

    # -- Inspect output and save the chain:
    samples = sampler.chain
    np.save(f'{output_dir}/{basename}_chains.npy', samples)

    # plt.close('all')
    # -- Plot chains:
    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    if ndim == 1:
        axes = [axes]

    for i in range(ndim):
        ax = axes[i]
        for num in range(nwalkers):
            ax.plot(samples[num, :, i], "k", alpha=0.3, lw=0.5)
        ax.set_xlim(0, steps)
        ax.set_ylabel(active_labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
        ax.axvline(burn_in, color='crimson', ls='--', lw=1.)
    axes[-1].set_xlabel("Step number")
    fig.tight_layout()
    fig.savefig(f"{output_dir}/{basename}_chains.pdf")

    # -- Get best-fit parameters and uncertainties:
    flat_samples = samples[:, burn_in:, :].reshape((-1, ndim))
    best_fit = np.median(flat_samples, 0)
    pars_1sig = np.percentile(flat_samples, [16., 84.], axis=0).T

    # Format into parameter dictionary
    best_fit_dict = {key: val for key, val in zip(variable_parnames, best_fit)}
    # Set up the parameters in fit_globals:
    fit_globals = {}
    for parname, parval in best_fit_dict.items():
        fit_globals[parname] = parval

    # Add fixed parameters to best-fit dictionary:
    for parname, par in parameters.items():
        if parname not in variable_parnames:
            if par.tie:
                # evaluate expression
                best_fit_dict[parname] = eval(par.tie, fit_globals)
            else:
                best_fit_dict[parname] = par.value


    # -- Save parameters:
    save_pars(f'{output_dir}/{basename}_pars.txt', best_fit_dict, variable_parnames)


    # -- Plot correlations and marginalized posterior PDFs:
    #    Include only the metallicity parameters, so only the first half of the array:
    if len(input_pars) > 0:
        truth_values = input_pars[:, 0]
        if not fix_depletion:
            truth_values = np.concatenate([truth_values, input_pars[:, 2]])
    else:
        truth_values = best_fit
    hist2d_kwargs = {'bins': 40}
    fig_corner = corner.corner(flat_samples,
                               labels=active_labels,
                               truths=truth_values,
                               truth_color='#4682b4' if len(input_pars) > 0 else 'crimson',
                               label_kwargs={'fontsize': 14},
                               smooth1d=0.1,
                                **hist2d_kwargs)

    # -- Write fit report:
    report.add_line("# Best-fit parameters:\n")
    fig_string = ''
    for i, label in enumerate(variable_parnames):
        med = best_fit[i]
        l68, u68 = pars_1sig[i]
        par_string = "%22s = %.2f +%.2f -%.2f\n" % (label, med, u68-med, med-l68)
        report.add_line(par_string)
        #if 'logZ' in label:
        fig_string += "%s = $%.2f ^{+%.2f} _{-%.2f}$" % (active_labels[i], med, u68-med, med-l68)
        if len(input_pars) > 0:
            if 'logZ' in label:
                fig_string += "(True: %.2f)" % input_pars[i][0]
            else:
                fig_string += "(True: %.2f)" % input_pars[i][2]
        fig_string += '\n'

    if len(input_pars) > 0:
        fig_string += '\nTrue parameters shown in blue'
    else:
        fig_string += '\Best-fit parameters shown in red'


    report.add_line("\n \nLaTeX Notation:\n")
    report.add_line(fig_string)

    # -- Write fit report and print to screen
    report.write(f'{output_dir}/{basename}_report.txt')
    # report.print_report()

    # -- Add best-fit solutions to the corner plot and save:
    fig_corner.text(0.95, 0.95, fig_string, va='top', ha='right', transform=fig_corner.transFigure)
    fig_corner.savefig(f"{output_dir}/{basename}.pdf")


def main():

    parser = ArgumentParser('Depletion Pattern Simulator')
    parser.add_argument("input", type=str,
                        help="Filename of the input file containing column densities")
    parser.add_argument("-o", "--output", type=str, default='output',
                        help="Output directory")
    parser.add_argument("-s", "--steps", type=int, default=5000,
                        help="Number of steps for the MCMC ensemble")
    parser.add_argument("-b", "--burn", type=int, default=500,
                        help="Number of steps to burn (before convergence)")
    parser.add_argument("-n", "--nwalkers", type=int, default=100,
                        help="Number of walkers in the ensemble")
    parser.add_argument("-w", "--weight", type=int, default=100,
                        help="Component weight for HI distribution (drawn random between 1 and `w`)")
    parser.add_argument("-f", "--fix", action='store_true',
                        help="Fix the depletion strength [Zn/Fe]_fit?")

    args = parser.parse_args()

    run_mcmc_sampler(args.input,
                     steps=args.steps,
                     burn_in=args.burn,
                     nwalkers=args.nwalkers,
                     H_WEIGHT=args.weight,
                     output_dir=args.output,
                     fix_depletion=args.fix,
                     )

if __name__ == '__main__':
    main()

