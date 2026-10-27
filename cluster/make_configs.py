"""Write one config file per arm of the second study into cluster/configs/.

Each line is the argument list of llm/run_llm.py for a single run. The RMU
setting is the one selected by the screen (lowest utility cost on seed 0).
"""
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")
os.makedirs(OUT, exist_ok=True)
RMU = "--rmu_alpha 1000 --rmu_c 6 --rmu_layer 5"
METHODS = ["--method ga", "--method gagd", "--method npo", "--method npo_sam",
           f"--method rmu {RMU}", f"--method mcu {RMU} --mcu_k 8",
           "--method cir --un_lr 1e-5", "--method tar", "--method dfu",
           "--method none"]            # 'none' runs the attacks from the never-unlearned model
ARMS = {"tofu": ("--bench tofu --metrics --strong", 5),
        "muse": ("--bench muse --metrics", 5),
        "wmdp": ("--bench wmdp --metrics --strong", 3),
        "tofu_fulladv": ("--bench tofu --adv_full", 3)}
for arm, (flags, seeds) in ARMS.items():
    lines = [f"{flags} {m} --seed {s}" for s in range(seeds) for m in METHODS]
    if arm == "wmdp":       # the reference attack is run inside every WMDP job
        lines = [l for l in lines if "--method none" not in l]
    with open(os.path.join(OUT, f"{arm}.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"{arm}: {len(lines)} runs")
screen = [f"--bench tofu --method rmu --rmu_alpha {a} --rmu_c {c} --rmu_layer {l} --seed 0"
          for a in (1, 100, 1000) for c in (6, 20) for l in (5, 15) if not (a == 1 and (c, l) != (6, 15))]
with open(os.path.join(OUT, "rmu_screen.txt"), "w") as fh:
    fh.write("\n".join(screen) + "\n")
print(f"rmu_screen: {len(screen)} runs")
