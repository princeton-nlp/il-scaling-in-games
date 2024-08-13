def update_mean(prev_m, prev_samples, new_m, new_samples):
    return (prev_m * prev_samples + new_m * new_samples)/(prev_samples + new_samples)