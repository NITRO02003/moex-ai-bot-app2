
def main(args):
    from .data_pipeline import run_data_processing
    run_data_processing(
        symbols=args.symbols,
        intervals=args.intervals,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        out=args.out
    )
