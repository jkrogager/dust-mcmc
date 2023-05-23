"""
Simulate multiple random input datasets using the `simulate_data.py` script.
"""

from argparse import ArgumentParser

from simulate_data import make_random_components, simulate_depletion_pattern


def main():
    parser = ArgumentParser('Depletion Pattern Simulator')
    parser.add_argument("number", type=int, default=100,
                        help="Number of simulations to generate.")
    parser.add_argument("-n", "--ncomp", type=int, default=2,
                        help="Number of random components to generate [default=2]")
    parser.add_argument("-e", "--elements", type=str, nargs='+', default=[],
                        help="Filename of the generated column density table (CSV file)"
                             "Ex.: -e C Fe Si Zn")
    parser.add_argument("--dir", type=str, default='data',
                        help="Output directory [default=data/]")

    args = parser.parse_args()

    print(f"Simulating depletion data for {args.ncomp} components")
    print(f"Saving data to folder: {args.dir}")
    for _ in range(args.number):
        components = make_random_components(args.ncomp)
        tablehash = hash(str(components))
        ncomp = len(components)
        output_filename = f"N{ncomp}_data.{tablehash}.csv"
        simulate_depletion_pattern(components, output_filename,
                                   elements=args.elements,
                                   output_dir=args.dir,
                                   verbose=False)
    print(f"Finished simulating {args.number} datasets")

if __name__ == '__main__':
    main()

