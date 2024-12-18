from il_scale.utils.flop import FLOPCounter, FLOP_TO_STR
from il_scale.networks.atari_networks import SimpleCNN

FPS = 6e3
BATCH_SIZE = 1

flop_counter = FLOPCounter()
MODEL_TO_FLOP = {
    "1k": flop_counter.count_flops(SimpleCNN(w_scale=1))["total_flops"],
    "2k": flop_counter.count_flops(SimpleCNN(w_scale=3))["total_flops"],
    "5k": flop_counter.count_flops(SimpleCNN(w_scale=5))["total_flops"],
    "10k": flop_counter.count_flops(SimpleCNN(w_scale=8))["total_flops"],
    "20k": flop_counter.count_flops(SimpleCNN(w_scale=13))["total_flops"],
    "50k": flop_counter.count_flops(SimpleCNN(w_scale=23))["total_flops"],
    "100k": flop_counter.count_flops(SimpleCNN(w_scale=34))["total_flops"],
    "200k": flop_counter.count_flops(SimpleCNN(w_scale=50))["total_flops"],
    "500k": flop_counter.count_flops(SimpleCNN(w_scale=81))["total_flops"],
    "1M": flop_counter.count_flops(SimpleCNN(w_scale=117))["total_flops"],
    "2M": flop_counter.count_flops(SimpleCNN(w_scale=167))["total_flops"],
    "5M": flop_counter.count_flops(SimpleCNN(w_scale=266))["total_flops"],
}

for flop in FLOP_TO_STR.values():
    row = f"{flop:6} |"
    for model in MODEL_TO_FLOP.keys():
        row += f" {float(flop)/MODEL_TO_FLOP[model]/BATCH_SIZE:10.2e} |"
    print(row)

print("\n")

for flop in FLOP_TO_STR.values():
    row = f"{flop:6} |"
    for model in MODEL_TO_FLOP.keys():
        row += f" {float(flop)/MODEL_TO_FLOP[model]/FPS/60/60:10.1f} |"
    print(row)
