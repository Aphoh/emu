import argparse
import json
from pathlib import Path
import subprocess


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process some prompts.")
    parser.add_argument('--plist', type=str, help='Path to the list of prompts')
    parser.add_argument('--start', type=int, required=True, help='Offset within the prompts')
    parser.add_argument('--end', type=int, required=True, help='Offset within the prompts')
    parser.add_argument('--outdir', type=str, help='Output directory')
    parser.add_argument('--metadata', type=str, help='Metadata JSONL file')
    parser.add_argument('--prompts_per_run', type=int, default=3, help='Number of prompts to process per run')
    
    return parser.parse_known_args()

def main():
    args, other_args = parse_arguments()
    
    # Load prompts
    with open(args.plist, 'r') as f:
        prompts = f.readlines()
    
    # Load metadata
    with open(args.metadata, 'r') as f:
        metadata = f.readlines()
    
    assert len(prompts) == len(metadata), "Number of prompts and metadata entries must match"

    inds = list(range(len(prompts)))
    start: int = args.start
    prompts_per_run: int = args.prompts_per_run
    prompts, metadata, inds = prompts[start:args.end], metadata[start:args.end], inds[start:args.end]

    batches = []
    for i in range(0, len(prompts), args.prompts_per_run):
        batch = [prompts[i:i + prompts_per_run], metadata[i:i + prompts_per_run], inds[i:i + prompts_per_run]]
        batches.append(batch)

    for prompts, metas, inds in batches:
        out_dirs = [Path(args.outdir) / f"{ind:05}" for ind in inds]
        for out_dir, meta in zip(out_dirs, metas):
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / "metadata.jsonl", 'w') as f:
                f.write(meta)

        cmd_outdirs = [str(out_dir / "samples") for out_dir in out_dirs]
        cmd_prompts = [prompt.strip() for prompt in prompts]
        cmd = ["python3", "-m", "emu.run.batched"] + other_args
        for cmd_outdir in cmd_outdirs:
            cmd += ["-o", cmd_outdir]
        for cmd_prompt in cmd_prompts:
            cmd += ["-p", cmd_prompt]

        print("Running command:", cmd)
        subprocess.check_call(cmd)


        
    
    # Process prompts and metadata
    # (Add your processing logic here)
    
    # Save results to output directory
    # (Add your saving logic here)

if __name__ == "__main__":
    main()