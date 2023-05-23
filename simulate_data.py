from VoigtFit.utils.Asplund import solar
from VoigtFit.utils import depletion
import numpy as np
from collections import defaultdict
from astropy.table import Table
import os

depletion_sequence = depletion.coeffs


def make_random_components(N, logZ_range=(-3., 0.5), logH_range=(19., 21.5), delta_range=(0., 2.5)):
    """
    Create `N` randomly drawn components. The metallicity, H column and depletion strength
    are all drawn uniformly within the defined ranges.
    """
    logZ = np.random.uniform(logZ_range[0], logZ_range[1], N)
    logH = np.random.uniform(logH_range[0], logH_range[1], N)
    delta = np.random.uniform(delta_range[0], delta_range[1], N)
    comps = np.column_stack([logZ, logH, delta])
    return Table(comps, names=['logZ', 'logNH', '[Zn/Fe]'])


def simulate_depletion_pattern(components, output_filename, elements=[], err_min=0.04, err_max=0.1,
                               output_dir='data', verbose=True):
    """
    Simulate column densities for a given combination of metallicity,
    hydrogen column density and depletion strength ([Zn/Fe]).

    Parameters
    ----------
    components: `astropy.table.Table`
        Astropy table (or similar) with three column names:
        logZ, logH, [Zn/He]
        Each row of the table corresponds to one absorption component.

    output_filename: str
        Filename for the saved filename of the simulation.

    elements: list[str]
        List of elements to include in the simulation.
        By default, all elements in the compilation by Christina Konstantopoulou et al. (2023)
        is included. Ex: ['Fe', 'Si', 'Zn', 'S']

    err_min, err_max : float
        Minimum and maximum error for each log column density (units of dex).
        The uncertainty is drawn uniformly between [min_err:max_err].

    output_dir : str
        Directory to place the simulated column densities into.

    verbose : bool [default=True]
        Print status messages?

    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_filename = os.path.join(output_dir, output_filename)

    all_logN = defaultdict(list)
    all_logN_err = defaultdict(list)
    all_X = []
    for row in components:
        logZ = row['logZ']
        logNHI = row['logNH']
        delta = row['[Zn/Fe]']
        for X, (A2, B2) in depletion_sequence.items():
            if X not in solar:
                continue
            if len(elements) > 0 and X not in elements:
                continue
            X_sun, _ = solar[X]
            logN_X = logZ + logNHI + (X_sun - 12) + A2 + B2*delta
            err = np.random.uniform(err_min, err_max)
            logN_X += err * np.random.normal()
            all_logN[X].append(logN_X)
            all_logN_err[X].append(err)

    with open(output_filename, 'w') as f:
        # Write input parameters:
        f.write("# Input parameters:\n")
        f.write("# logZ, logNH, [Zn/Fe]\n")
        for row in components:
            f.write("# %.3f, %.3f, %.3f\n" % tuple(row))
        logNHI_tot = np.log10(np.sum(10**components['logNH']))
        header = "ion,total,total_err,"
        header += ",".join(["comp%i,comp%i_err" % (i+1, i+1) for i in range(len(components))])
        f.write(header + "\n")
        f.write(f"H,{logNHI_tot:.2f}, 0.05" + 2*len(components)*"," + "\n")

        for X, ln in all_logN.items():
            ln = np.array(ln)
            ln_err = np.array(all_logN_err[X])
            N_tot = np.sum(10**ln)
            err_tot = np.sqrt(np.sum(ln_err**2 * 10**ln)/N_tot)
            line = f"{X}, {np.log10(N_tot):.2f}, {err_tot:.2f}, "
            line += ', '.join(["%.2f, %.2f" % (l, e) for l, e in zip(ln, ln_err)])
            f.write(line+'\n')
    if verbose:
        print(f"Wrote simulated column densities for {len(components)} components to file:")
        print(f"{output_filename}")
        print("")


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser('Depletion Pattern Simulator')
    parser.add_argument("filename", nargs='?', type=str, default='',
                        help="Filename of input CSV file (optional)."
                        "If not given, a random set of N components are generated.")
    parser.add_argument("-n", "--number", type=int, default=2,
                        help="Number of random components to generate [default=2]")
    parser.add_argument("--Zmin", type=float, default=-1.5,
                        help="Minimum range of metallicity for random components [default=-1.5]")
    parser.add_argument("--Zmax", type=float, default=0.5,
                        help="Maximum range of metallicity for random components [default=0.5]")
    parser.add_argument("--Hmin", type=float, default=19.0,
                        help="Minimum range of log N(H) for random components [default=19.0]")
    parser.add_argument("--Hmax", type=float, default=21.5,
                        help="Maximum range of log N(H) for random components [default=21.5]")
    parser.add_argument("--dmin", type=float, default=0.,
                        help="Minimum range of [Zn/Fe] for random components [default=0.]")
    parser.add_argument("--dmax", type=float, default=2.5,
                        help="Maximum range of [Zn/Fe] for random components [default=2.5]")
    parser.add_argument("-o", "--output", type=str, default='',
                        help="Filename of the generated column density table (CSV file)"
                             "autogenerated by default")
    parser.add_argument("-e", "--elements", type=str, nargs='+', default=[],
                        help="Filename of the generated column density table (CSV file)"
                             "Ex.: -e C Fe Si Zn")
    parser.add_argument("--error-min", type=float, default=0.04,
                        help="Minimum random uncertainty on log(N) for each element")
    parser.add_argument("--error-max", type=float, default=0.10,
                        help="Maximum random uncertainty on log(N) for each element")

    args = parser.parse_args()

    if len(args.filename) == 0:
        Zmin = args.Zmin
        Zmax = args.Zmax
        Hmin = args.Hmin
        Hmax = args.Hmax
        dmin = args.dmin
        dmax = args.dmax
        components = make_random_components(args.number,
                                            logZ_range=(Zmin, Zmax),
                                            logH_range=(Hmin, Hmax),
                                            delta_range=(dmin, dmax),
                                            )
    else:
        components = Table.read(args.filename, comment='#')

    if args.output:
        output_filename = f"{output_dir}/" + args.output
        if not output_filename.endswith('.csv'):
            output_filename += '.csv'
    else:
        tablehash = hash(str(components))
        ncomp = len(components)
        output_filename = f"N{ncomp}_data.{tablehash}.csv"

    simulate_depletion_pattern(components, output_filename, elements=args.elements)


if __name__ == '__main__':
    main()

