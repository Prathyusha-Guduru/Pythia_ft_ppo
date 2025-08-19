import subprocess, os, pathlib
import time

GAMMA = [0.5, 0.95, 1.0]
LAM   = [0.5, 0.95, 1.0]

ACCEL_CFG = "../../accelerate_configs/deepspeed_zero2.yaml"
BASE_OUT  = "/scratch/cluster/piti/trl/examples/scripts/ppo/aug_18"

def slug(x: float) -> str:
    return str(x).replace(".", "p")

ENV = os.environ.copy()
ENV.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

pathlib.Path(BASE_OUT).mkdir(parents=True, exist_ok=True)

MAX_RETRIES = 5  
SLEEP_SEC   = 60 

midi_3_pairs = [(0.5, 1.0) ,(0.95, 1.0)]
midi_2_pairs = [(1.0, 0.95), (1.0, 1.0)]

done_pairs = [(0.5, 0.5),
            (0.5, 0.95),
            (0.95, 0.5),
            (0.95, 0.95),
            (1.0, 0.5)]

for g in GAMMA:
    for l in LAM:
        # Skip  the existing models
        if (g, l) in done_pairs or (g,l) in midi_3_pairs:
            print(f"Skipping run for gamma={g}, lam={l}")
            continue

        run_name   = f"tldr-ppco-g{slug(g)}-l{slug(l)}"
        output_dir = f"{BASE_OUT}/{run_name}"
        log_path   = f"{output_dir}.log"

        cmd = [
            "accelerate", "launch", "--config_file", ACCEL_CFG,
            "ppo_tldr.py",
            "--dataset_name", "trl-internal-testing/tldr-preference-sft-trl-style",
            "--dataset_test_split", "validation",
            "--output_dir", output_dir,
            "--per_device_train_batch_size", "1",
            "--gradient_accumulation_steps", "4",
            "--learning_rate", "5e-5",
            "--model_name_or_path", "EleutherAI/pythia-1b-deduped",
            "--sft_model_path", "cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr",
            "--reward_model_path", "cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr",
            "--missing_eos_penalty", "1.0",
            "--stop_token", "eos",
            "--response_length", "52",
            "--eval_strategy", "steps",
            "--eval_steps", "100",
            "--total_episodes", "40000",
            "--num_ppo_epochs", "4",
            "--push_to_hub", "True",
            "--gamma", str(g),
            "--lam", str(l),
            "--run_name", run_name,           # if your script supports this arg
        ]

        # print(f"\n=== Launching {run_name} ===")
        # pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        # with open(log_path, "w") as lf:
        #     lf.write(f"# CMD: {' '.join(cmd)}\n\n")
        #     lf.flush()
        #     # block until the run fully completes
        #     subprocess.run(cmd, check=True, env=ENV, stdout=lf, stderr=lf)
        # print(f"✅ Finished {run_name}. Logs: {log_path}")
        # # optional: remove the output bin file to save space
        # bin_file = os.path.join(output_dir, "pytorch_model.bin")

        # # Remove only the .bin file if it exists
        # if os.path.exists(bin_file):
        #     subprocess.run(["rm", "-f", bin_file], check=True)
        #     print(f"🗑️  Removed file: {bin_file}")
        # else:
        #     print("⚠️  pytorch_model.bin not found.")

        print(f"\n=== Launching {run_name} ===")
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

        success = False
        for attempt in range(1, MAX_RETRIES + 1):
            with open(log_path, "a") as lf:  # append so retries go in same log
                lf.write(f"\n# Attempt {attempt}: {' '.join(cmd)}\n\n")
                lf.flush()
                try:
                    subprocess.run(cmd, check=True, env=ENV, stdout=lf, stderr=lf)
                    success = True
                    print(f"✅ Finished {run_name} on attempt {attempt}. Logs: {log_path}")
                    break
                except subprocess.CalledProcessError as e:
                    print(f"❌ {run_name} failed (attempt {attempt}/{MAX_RETRIES}): {e}")
                    if attempt < MAX_RETRIES:
                        print(f"⏳ Retrying in {SLEEP_SEC}s…")
                        time.sleep(SLEEP_SEC)
        if not success:
            print(f"🚨 {run_name} failed after {MAX_RETRIES} attempts, skipping…")
            continue

        # optional cleanup
        bin_file = os.path.join(output_dir, "pytorch_model.bin")
        if os.path.exists(bin_file):
            os.remove(bin_file)
            print(f"🗑️  Removed file: {bin_file}")
        else:
            print("⚠️  pytorch_model.bin not found.")
