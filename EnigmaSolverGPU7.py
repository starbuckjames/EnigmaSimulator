from collections import Counter, defaultdict
from itertools import permutations, combinations
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# Numba and Numpy are required for GPU acceleration
try:
    import numpy as np
    from numba import cuda
except ImportError:
    pass # Will be handled in main execution block

#Enigma Machine Solver using GPU parallelization with Numba and CUDA

rotor_1_string = "EKMFLGDQVZNTOWYHXUSPAIBRCJ"
rotor_2_string = "AJDKSIRUXBLHWTMCQGZNPYFVOE"
rotor_3_string = "BDFHJLCPRTXVZNYEIWGAKMUSQO"
rotor_4_string = "ESOVPZJAYQUIRHXLNFTGKDCMWB"
rotor_5_string = "VZBRGITYUPSDNHLXAWMJQOFECK"

etw_string = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
reflector_b_string = "YRUHQSLDPXNGOKMIEBFZCWVJAT"
reflector_c_string = "FVPJIAOYEDRZXWGCTKUQSBNMHL"

rotor_1_array = list(rotor_1_string)
rotor_2_array = list(rotor_2_string)
rotor_3_array = list(rotor_3_string)
rotor_4_array = list(rotor_4_string)
rotor_5_array = list(rotor_5_string)

etw_array = list(etw_string)
reflector_b_array = list(reflector_b_string)
reflector_c_array = list(reflector_c_string)

rotor_1_notch = 16
rotor_2_notch = 4
rotor_3_notch = 21
rotor_4_notch = 9
rotor_5_notch = 25

rotor_dict = {
    "1": rotor_1_array,
    "2": rotor_2_array,
    "3": rotor_3_array,
    "4": rotor_4_array,
    "5": rotor_5_array
}

# English Bigram Log-Probability Scores (Top 150+ common combinations)
# A score of -1.36 means the bigram occurs ~4% of the time.
# A score of -4.00 means it occurs 0.01% of the time.
BIGRAM_SCORES = {
    "TH": -1.36, "HE": -1.48, "IN": -1.53, "ER": -1.59, "AN": -1.62,
    "RE": -1.65, "ON": -1.71, "AT": -1.73, "EN": -1.74, "ND": -1.78,
    "TI": -1.81, "ES": -1.82, "OR": -1.84, "TE": -1.85, "OF": -1.87,
    "ED": -1.91, "IS": -1.93, "IT": -1.95, "AL": -1.97, "AR": -1.99,
    "ST": -2.01, "TO": -2.03, "NT": -2.05, "NG": -2.07, "SE": -2.09,
    "HA": -2.11, "AS": -2.12, "OU": -2.13, "IO": -2.14, "LE": -2.15,
    "VE": -2.17, "CO": -2.19, "ME": -2.21, "DE": -2.23, "HI": -2.24,
    "RI": -2.26, "RO": -2.28, "IC": -2.30, "NE": -2.32, "EA": -2.34,
    "RA": -2.35, "CE": -2.37, "LI": -2.39, "CH": -2.41, "LL": -2.43,
    "BE": -2.45, "MA": -2.47, "SI": -2.49, "OM": -2.51, "UR": -2.53,
    "AC": -2.55, "OT": -2.57, "EE": -2.59, "IL": -2.61, "ID": -2.63,
    "ET": -2.65, "LO": -2.67, "UT": -2.69, "ST": -2.71, "CK": -2.73,
    "QU": -2.75, "GE": -2.77, "IR": -2.79, "SH": -2.81, "LA": -2.83,
    "PR": -2.85, "PE": -2.87, "AI": -2.89, "AD": -2.91, "IF": -2.93,
    "NS": -2.95, "PO": -2.97, "TA": -2.99, "DA": -3.01, "TY": -3.03,
    "HO": -3.05, "FO": -3.07, "PA": -3.09, "EC": -3.11, "KE": -3.13,
    "RS": -3.15, "LY": -3.17, "UL": -3.19, "IM": -3.21, "MI": -3.23,
    "CA": -3.25, "TR": -3.27, "AM": -3.29, "UN": -3.31, "SO": -3.33,
    "WI": -3.35, "OW": -3.37, "PI": -3.39, "GE": -3.41, "NC": -3.43,
    "PL": -3.45, "MO": -3.47, "DI": -3.49, "GA": -3.51, "FE": -3.53,
    "LU": -3.55, "US": -3.57, "NA": -3.59, "SA": -3.61, "UR": -3.63,
    "BY": -3.65, "WE": -3.67, "NO": -3.69, "NI": -3.71, "DO": -3.73,
}

# Anything not in the table is likely very rare (like 'QX' or 'ZJ')
BIGRAM_FLOOR = -4.5

# --- PRECOMPUTED INTEGER MAPPINGS FOR GPU ---
# Convert all character-based settings to integer arrays (0-25) for the GPU.
# This avoids slow string/dict operations in the kernels.
ROTOR_FWD_INT = {k: [ord(c)-65 for c in v] for k, v in rotor_dict.items()}
ROTOR_REV_INT = {}
for k, v in ROTOR_FWD_INT.items():
    rev = [0]*26
    for i, c in enumerate(v): rev[c] = i
    ROTOR_REV_INT[k] = rev

ROTOR_NOTCH_INT = { "1": 16, "2": 4, "3": 21, "4": 9, "5": 25 }
REFLECTOR_B_INT = [ord(c)-65 for c in reflector_b_string]


def calculate_bigram(text):
    """Calculates the bigram score of a text, a measure of how 'English-like' it is."""
    text = "".join(filter(str.isalpha, text.upper()))
    if len(text) < 2:
        return -999.0

    total_score = 0
    for i in range(len(text) - 1):
        bigram = text[i:i+2]
        total_score += BIGRAM_SCORES.get(bigram, BIGRAM_FLOOR)

    return total_score

def encrypt_message_with_pb(message, r_start, m_start, l_start, r_rotor, m_rotor, l_rotor, plugboard_dict):
    """
    A CPU-based Enigma encryption function for verification and final decryption.
    This version includes the authentic double-stepping mechanism and uses integer-based
    logic for consistency with the GPU kernels we will build.
    """
    message = message.upper()
    message_encrypted = ""
    r_pos, m_pos, l_pos = r_start, m_start, l_start

    # Map rotors to their names and notch positions
    rotor_map = {tuple(v): k for k, v in rotor_dict.items()}
    r_name, m_name, l_name = rotor_map[tuple(r_rotor)], rotor_map[tuple(m_rotor)], rotor_map[tuple(l_rotor)]
    r_notch, m_notch = ROTOR_NOTCH_INT[r_name], ROTOR_NOTCH_INT[m_name]

    # Pre-fetch integer representations for speed
    r_fwd, m_fwd, l_fwd = ROTOR_FWD_INT[r_name], ROTOR_FWD_INT[m_name], ROTOR_FWD_INT[l_name]
    r_rev, m_rev, l_rev = ROTOR_REV_INT[r_name], ROTOR_REV_INT[m_name], ROTOR_REV_INT[l_name]

    # Convert plugboard to integer representation
    pb_int = {ord(k)-65: ord(v)-65 for k,v in plugboard_dict.items()}
    # Add reverse mappings
    pb_int.update({v: k for k, v in pb_int.items()})

    for char_in in message:
        if 'A' <= char_in <= 'Z':
            # 1. Rotor stepping (with authentic double-stepping)
            right_at_notch = (r_pos == r_notch)
            middle_at_notch = (m_pos == m_notch)

            r_pos = (r_pos + 1) % 26
            if right_at_notch or middle_at_notch:
                m_pos = (m_pos + 1) % 26
                if middle_at_notch:
                    l_pos = (l_pos + 1) % 26

            # 2. Encryption Path (using integer logic)
            letter = ord(char_in) - 65
            letter = pb_int.get(letter, letter)

            # Forward path
            idx = (letter + r_pos) % 26; letter = (r_fwd[idx] - r_pos + 26) % 26
            idx = (letter + m_pos) % 26; letter = (m_fwd[idx] - m_pos + 26) % 26
            idx = (letter + l_pos) % 26; letter = (l_fwd[idx] - l_pos + 26) % 26

            # Reflector
            letter = REFLECTOR_B_INT[letter]

            # Reverse path
            idx = (letter + l_pos) % 26; letter = (l_rev[idx] - l_pos + 26) % 26
            idx = (letter + m_pos) % 26; letter = (m_rev[idx] - m_pos + 26) % 26
            idx = (letter + r_pos) % 26; letter = (r_rev[idx] - r_pos + 26) % 26

            letter = pb_int.get(letter, letter)
            message_encrypted += chr(letter + 65)
        else:
            message_encrypted += char_in # Pass non-alpha characters through

    return message_encrypted

def pre_step_rotors(r_start, m_start, l_start, offset, r_name, m_name):
    """
    Calculates the rotor positions after a given number of steps (offset).
    This is a CPU-side helper to pre-compute work for the GPU.
    """
    r_notch = ROTOR_NOTCH_INT[r_name]
    m_notch = ROTOR_NOTCH_INT[m_name]

    r_pos, m_pos, l_pos = r_start, m_start, l_start

    for _ in range(offset):
        right_at_notch = (r_pos == r_notch)
        middle_at_notch = (m_pos == m_notch)

        r_pos = (r_pos + 1) % 26
        if right_at_notch or middle_at_notch:
            m_pos = (m_pos + 1) % 26
            if middle_at_notch:
                l_pos = (l_pos + 1) % 26

    return r_pos, m_pos, l_pos
# --- GPU BOMBE KERNELS AND HELPERS (for Numba) ---

@cuda.jit(device=True)
def core_encrypt_int_gpu(letter, r_pos, m_pos, l_pos, r_fwd, r_rev, m_fwd, m_rev, l_fwd, l_rev, ref):
    """GPU device version of the core encryption logic (no plugboard)."""
    idx = (letter + r_pos) % 26; letter = (r_fwd[idx] - r_pos + 26) % 26
    idx = (letter + m_pos) % 26; letter = (m_fwd[idx] - m_pos + 26) % 26
    idx = (letter + l_pos) % 26; letter = (l_fwd[idx] - l_pos + 26) % 26
    letter = ref[letter]
    idx = (letter + l_pos) % 26; letter = (l_rev[idx] - l_pos + 26) % 26
    idx = (letter + m_pos) % 26; letter = (m_rev[idx] - m_pos + 26) % 26
    idx = (letter + r_pos) % 26; letter = (r_rev[idx] - r_pos + 26) % 26
    return letter

@cuda.jit(device=True)
def backtrack_bombe_gpu(pair_idx_start, pb_out, positions, pairs, r_fwd, r_rev, m_fwd, m_rev, l_fwd, l_rev, ref, crib_len):
    # INCREASED limit and refined stack to prevent overflow and logic errors
    CRIB_MAX_LEN = 32 
    CRIB_STACK_DEPTH = 33

    pb_stack = cuda.local.array((CRIB_STACK_DEPTH, 26), dtype=np.int8)
    plugged_vals_stack = cuda.local.array((CRIB_STACK_DEPTH, 26), dtype=np.bool_)
    iterator_stack = cuda.local.array(CRIB_MAX_LEN, dtype=np.int8)

    # Init level 0
    for i in range(26):
        pb_stack[0, i] = -1
        plugged_vals_stack[0, i] = False
    for i in range(CRIB_MAX_LEN): iterator_stack[i] = 0

    pair_idx = 0
    while pair_idx >= 0:
        if pair_idx == crib_len:
            for i in range(26): pb_out[i] = pb_stack[crib_len, i]
            return True

        pb = pb_stack[pair_idx]
        plugged_vals = plugged_vals_stack[pair_idx]
        p_char, c_char = pairs[pair_idx]
        r_pos, m_pos, l_pos = positions[pair_idx]
        
        p_val, c_val = pb[p_char], pb[c_char]
        advanced = False

        if p_val != -1 or c_val != -1: # Deterministic
            if iterator_stack[pair_idx] == 0:
                iterator_stack[pair_idx] = 1
                for i in range(26):
                    pb_stack[pair_idx + 1, i] = pb[i]
                    plugged_vals_stack[pair_idx + 1, i] = plugged_vals[i]
                
                next_pb = pb_stack[pair_idx + 1]
                next_pv = plugged_vals_stack[pair_idx + 1]

                if p_val != -1:
                    target_c = core_encrypt_int_gpu(p_val, r_pos, m_pos, l_pos, r_fwd, r_rev, m_fwd, m_rev, l_fwd, l_rev, ref)
                    if next_pb[c_char] == target_c:
                        pair_idx += 1; advanced = True
                    elif next_pb[c_char] == -1 and not next_pv[target_c]:
                        next_pb[c_char], next_pb[target_c] = target_c, c_char
                        next_pv[target_c], next_pv[c_char] = True, True
                        pair_idx += 1; advanced = True
                else: # c_val != -1
                    target_p = core_encrypt_int_gpu(c_val, r_pos, m_pos, l_pos, r_fwd, r_rev, m_fwd, m_rev, l_fwd, l_rev, ref)
                    if next_pb[p_char] == target_p:
                        pair_idx += 1; advanced = True
                    elif next_pb[p_char] == -1 and not next_pv[target_p]:
                        next_pb[p_char], next_pb[target_p] = target_p, p_char
                        next_pv[target_p], next_pv[p_char] = True, True
                        pair_idx += 1; advanced = True
        else: # Guessing
            for guess_p in range(iterator_stack[pair_idx], 26):
                if plugged_vals[guess_p]: continue
                
                target_c = core_encrypt_int_gpu(guess_p, r_pos, m_pos, l_pos, r_fwd, r_rev, m_fwd, m_rev, l_fwd, l_rev, ref)
                
                # Simple contradiction check
                if plugged_vals[target_c] and target_c != guess_p: continue
                
                # Copy state
                for i in range(26):
                    pb_stack[pair_idx + 1, i] = pb[i]
                    plugged_vals_stack[pair_idx + 1, i] = plugged_vals[i]
                
                next_pb = pb_stack[pair_idx + 1]
                next_pv = plugged_vals_stack[pair_idx + 1]
                
                # Apply guess
                next_pb[p_char], next_pb[guess_p] = guess_p, p_char
                next_pv[p_char], next_pv[guess_p] = True, True
                
                # Check for c_char (which is now also deterministic)
                if next_pb[c_char] == target_c:
                    iterator_stack[pair_idx] = guess_p + 1
                    pair_idx += 1; advanced = True; break
                elif next_pb[c_char] == -1 and not next_pv[target_c]:
                    next_pb[c_char], next_pb[target_c] = target_c, c_char
                    next_pv[c_char], next_pv[target_c] = True, True
                    iterator_stack[pair_idx] = guess_p + 1
                    pair_idx += 1; advanced = True; break

        if not advanced:
            iterator_stack[pair_idx] = 0
            pair_idx -= 1
    return False

@cuda.jit
def bombe_kernel(tasks, msg_d, crib_d, reorder_idx_d, all_rf, all_rr, all_rn, ref_d, out_pbs_d, crib_len):
    """GPU kernel for the Bombe. Each thread handles one full setting combination."""
    task_id = cuda.grid(1)
    if task_id >= len(tasks): return

    # Task now contains pre-stepped positions; original starts (r_s, m_s, l_s) are ignored by kernel
    r_idx, m_idx, l_idx, r_s, m_s, l_s, r_offset_start, m_offset_start, l_offset_start, offset = tasks[task_id]

    r_fwd, m_fwd, l_fwd = all_rf[r_idx], all_rf[m_idx], all_rf[l_idx]
    r_rev, m_rev, l_rev = all_rr[r_idx], all_rr[m_idx], all_rr[l_idx]
    r_notch, m_notch = all_rn[r_idx], all_rn[m_idx]

    CRIB_MAX_LEN_KERNEL = 21 # IMPORTANT: Must be >= len(crib) and match in backtrack_bombe_gpu

    # --- Inside-thread setup ---
    pairs = cuda.local.array((CRIB_MAX_LEN_KERNEL, 2), dtype=np.int8)
    positions = cuda.local.array((CRIB_MAX_LEN_KERNEL, 3), dtype=np.int8)
    pb = cuda.local.array(26, dtype=np.int8)
    sequential_positions = cuda.local.array((CRIB_MAX_LEN_KERNEL, 3), dtype=np.int8)

    # Start from the pre-stepped positions calculated on the CPU
    curr_r, curr_m, curr_l = r_offset_start, m_offset_start, l_offset_start

    # OPTIMIZED: This loop now only runs crib_len times, not offset + crib_len
    # First, calculate all positions for the crib's length sequentially
    for i in range(crib_len):
        right_at_notch = (curr_r == r_notch)
        middle_at_notch = (curr_m == m_notch)
        curr_r = (curr_r + 1) % 26
        if right_at_notch or middle_at_notch:
            curr_m = (curr_m + 1) % 26
            if middle_at_notch:
                curr_l = (curr_l + 1) % 26
        sequential_positions[i, 0], sequential_positions[i, 1], sequential_positions[i, 2] = curr_r, curr_m, curr_l

    # Second, build the reordered pairs and positions arrays for the solver
    for i in range(crib_len):
        original_idx = reorder_idx_d[i]
        # Assign the correctly calculated position based on the original index
        positions[i, 0] = sequential_positions[original_idx, 0]
        positions[i, 1] = sequential_positions[original_idx, 1]
        positions[i, 2] = sequential_positions[original_idx, 2]
        # Create the reordered pairs
        pairs[i, 0] = crib_d[original_idx]
        pairs[i, 1] = msg_d[offset + original_idx]

    # Initialize the output plugboard for this thread to -1
    for i in range(26): pb[i] = -1

    if backtrack_bombe_gpu(0, pb, positions, pairs, r_fwd, r_rev, m_fwd, m_rev, l_fwd, l_rev, ref_d, crib_len):
        # If successful, write the found plugboard to global memory
        for i in range(26):
            out_pbs_d[task_id, i] = pb[i]

def run_bombe_phase_gpu(tasks_cpu, message_ints, crib_ints, message_str):
    """Host-side function to manage the GPU-based Bombe with progress reporting."""
    # 1. Flatten all tasks for the GPU
    flat_tasks = []
    # Create a mapping from rotor name (string) to a unique integer index (0-4)
    rotor_map = {k: i for i, k in enumerate(sorted(rotor_dict.keys()))}
    for arrangement, r_s, m_s, l_s, r_offset, m_offset, l_offset, offset in tasks_cpu:
        # Map string names to integer indices
        r_i, m_i, l_i = rotor_map[arrangement[2]], rotor_map[arrangement[1]], rotor_map[arrangement[0]]
        flat_tasks.append((r_i, m_i, l_i, r_s, m_s, l_s, r_offset, m_offset, l_offset, offset))

    tasks_np = np.array(flat_tasks, dtype=np.int32)
    total_tasks = len(tasks_np)

    # 2. Prepare STATIC data and transfer to GPU (once)
    print(f"Transferring static data to GPU memory...")
    msg_d = cuda.to_device(np.array(message_ints, dtype=np.int8))
    crib_d = cuda.to_device(np.array(crib_ints, dtype=np.int8))
    ref_d = cuda.to_device(np.array(REFLECTOR_B_INT, dtype=np.int32))

    # Create numpy arrays of all rotor wirings/notches, sorted by rotor name
    sorted_keys = sorted(ROTOR_FWD_INT.keys())
    all_rf = np.array([ROTOR_FWD_INT[k] for k in sorted_keys], dtype=np.int32)
    all_rr = np.array([ROTOR_REV_INT[k] for k in sorted_keys], dtype=np.int32)
    all_rn = np.array([ROTOR_NOTCH_INT[k] for k in sorted_keys], dtype=np.int32)
    all_rf_d, all_rr_d, all_rn_d = cuda.to_device(all_rf), cuda.to_device(all_rr), cuda.to_device(all_rn)

    # Pre-calculate the reordering index for the crib to speed up backtracking
    char_indices = defaultdict(list)
    for i, char_code in enumerate(crib_ints): char_indices[char_code].append(i)
    sorted_chars = sorted(char_indices.keys(), key=lambda c: len(char_indices[c]), reverse=True)
    reorder_idx = []; [reorder_idx.extend(char_indices[c]) for c in sorted_chars]
    reorder_idx_d = cuda.to_device(np.array(reorder_idx, dtype=np.int32))

    # 3. Process tasks in chunks to provide progress feedback
    chunk_size = 256 * 2048  # Process ~524k tasks per chunk
    candidates = []
    rev_rotor_map = {i: k for k, i in rotor_map.items()}

    print(f"Launching Bombe kernel in chunks of {chunk_size:,}...")
    for i in tqdm(range(0, total_tasks, chunk_size), desc="GPU Bombe Progress", unit="chunk"):
        # A. Get the current chunk of tasks
        start_idx = i
        end_idx = min(i + chunk_size, total_tasks)
        task_chunk_np = tasks_np[start_idx:end_idx]
        current_chunk_size = len(task_chunk_np)

        if current_chunk_size == 0: continue

        # B. Transfer dynamic data (tasks chunk) and allocate output buffer
        tasks_d = cuda.to_device(task_chunk_np)
        out_pbs_d = cuda.device_array((current_chunk_size, 26), dtype=np.int8)
        out_pbs_d.copy_to_device(np.full((current_chunk_size, 26), -1, dtype=np.int8))

        # C. Launch Kernel for the chunk
        threads_per_block = 256
        blocks_per_grid = (current_chunk_size + (threads_per_block - 1)) // threads_per_block
        bombe_kernel[blocks_per_grid, threads_per_block](
            tasks_d, msg_d, crib_d, reorder_idx_d, all_rf_d, all_rr_d, all_rn_d, ref_d, out_pbs_d, len(crib_ints)
        )
        cuda.synchronize()

        # D. Process results from the chunk
        out_pbs_h = out_pbs_d.copy_to_host()
        hit_indices_in_chunk = np.where(out_pbs_h[:, 0] != -1)[0]

        for hit_idx in hit_indices_in_chunk:
            pb_int = out_pbs_h[hit_idx]
            r_i, m_i, l_i, r_s, m_s, l_s, r_off, m_off, l_off, offset = task_chunk_np[hit_idx]

            arrangement = (rev_rotor_map[l_i], rev_rotor_map[m_i], rev_rotor_map[r_i])
            pb_dict_int = {j: pb_int[j] for j in range(26) if pb_int[j] != -1 and j < pb_int[j]}
            char_pb = {chr(k + 65): chr(v + 65) for k, v in pb_dict_int.items()}

            test_text = encrypt_message_with_pb(message_str, r_s, m_s, l_s, rotor_dict[arrangement[2]], rotor_dict[arrangement[1]], rotor_dict[arrangement[0]], char_pb)
            score = calculate_bigram(test_text)
            candidates.append({"score": score, "settings": (arrangement, r_s, m_s, l_s), "offset": offset, "partial_pb": pb_dict_int, "text": test_text})

    print("All chunks processed.")
    return candidates

# --- GPU HILL CLIMB KERNELS AND HELPERS ---

@cuda.jit(device=True)
def encrypt_message_gpu(msg_in, msg_out, r_s, m_s, l_s, r_f, r_r, m_f, m_r, l_f, l_r, r_n, m_n, ref, pb):
    """GPU device function to encrypt an entire message with a full plugboard."""
    r_p, m_p, l_p = r_s, m_s, l_s
    for i in range(len(msg_in)):
        right_at_notch = (r_p == r_n); middle_at_notch = (m_p == m_n)
        r_p = (r_p + 1) % 26
        if right_at_notch or middle_at_notch:
            m_p = (m_p + 1) % 26
            if middle_at_notch: l_p = (l_p + 1) % 26

        letter = pb[msg_in[i]]
        letter = core_encrypt_int_gpu(letter, r_p, m_p, l_p, r_f, r_r, m_f, m_r, l_f, l_r, ref)
        msg_out[i] = pb[letter]

@cuda.jit(device=True)
def calculate_bigram_gpu(text, bigram_scores_d, floor_score):
    """GPU device function to calculate bigram score."""
    score = 0.0
    for i in range(len(text) - 1):
        c1, c2 = text[i], text[i+1]
        score += bigram_scores_d[c1, c2]
    return score

@cuda.jit
def hill_climb_kernel(msg_in, unplugged_pairs, current_pb, r_s, m_s, l_s,
                      r_f, r_r, m_f, m_r, l_f, l_r, r_n, m_n, ref,
                      bigram_scores, floor_score, out_scores, scratchpad, msg_len):
    """GPU Kernel: each thread tests one possible new plugboard pair."""
    idx = cuda.grid(1)
    if idx >= len(unplugged_pairs): return

    # 1. Build this thread's test plugboard in local memory
    test_pb = cuda.local.array(26, dtype=np.int8)
    for i in range(26): test_pb[i] = current_pb[i]
    l1, l2 = unplugged_pairs[idx]
    test_pb[l1], test_pb[l2] = l2, l1

    # 2. Decrypt the message into this thread's slice of the scratchpad
    start = idx * msg_len
    decrypted_text_slice = scratchpad[start : start + msg_len]
    encrypt_message_gpu(msg_in, decrypted_text_slice, r_s, m_s, l_s, r_f, r_r, m_f, m_r, l_f, l_r, r_n, m_n, ref, test_pb)

    # 3. Score the result and write to output
    out_scores[idx] = calculate_bigram_gpu(decrypted_text_slice, bigram_scores, floor_score)

def run_hill_climb_phase_gpu(candidates, message_ints, original_message_with_spaces):
    """Host-side function to manage the GPU-based hill climb."""
    # 1. Prepare static data for GPU (once)
    msg_len = len(message_ints)
    msg_d = cuda.to_device(np.array(message_ints, dtype=np.int8))
    ref_d = cuda.to_device(np.array(REFLECTOR_B_INT, dtype=np.int32))

    bigram_np = np.full((26, 26), BIGRAM_FLOOR, dtype=np.float32)
    for k, v in BIGRAM_SCORES.items(): bigram_np[ord(k[0])-65, ord(k[1])-65] = v
    bigram_d = cuda.to_device(bigram_np)

    rotor_map = {k: i for i, k in enumerate(sorted(rotor_dict.keys()))}
    sorted_keys = sorted(ROTOR_FWD_INT.keys())
    all_rf_d = cuda.to_device(np.array([ROTOR_FWD_INT[k] for k in sorted_keys], dtype=np.int32))
    all_rr_d = cuda.to_device(np.array([ROTOR_REV_INT[k] for k in sorted_keys], dtype=np.int32))
    all_rn_d = cuda.to_device(np.array([ROTOR_NOTCH_INT[k] for k in sorted_keys], dtype=np.int32))

    fully_solved = []
    for cand in tqdm(candidates, desc="GPU Hill Climb", unit="candidate"):
        arrangement, r_s, m_s, l_s = cand["settings"]
        r_i, m_i, l_i = rotor_map[arrangement[2]], rotor_map[arrangement[1]], rotor_map[arrangement[0]]

        current_pb = np.arange(26, dtype=np.int8)
        for k, v in cand["partial_pb"].items(): current_pb[k], current_pb[v] = v, k
        best_score = cand["score"]

        # 2. Iteratively find the best plugboard pair to add
        while True:
            unplugged_ints = [i for i in range(26) if current_pb[i] == i]
            if len(unplugged_ints) < 2: break

            unplugged_pairs = np.array(list(combinations(unplugged_ints, 2)), dtype=np.int8)
            num_pairs = len(unplugged_pairs)

            # 3. Launch kernel to test all pairs in parallel
            current_pb_d = cuda.to_device(current_pb)
            unplugged_pairs_d = cuda.to_device(unplugged_pairs)
            out_scores_d = cuda.device_array(num_pairs, dtype=np.float32)
            scratchpad_d = cuda.device_array(num_pairs * msg_len, dtype=np.int8)

            threads_per_block = 128
            blocks_per_grid = (num_pairs + threads_per_block - 1) // threads_per_block

            hill_climb_kernel[blocks_per_grid, threads_per_block](
                msg_d, unplugged_pairs_d, current_pb_d, r_s, m_s, l_s,
                all_rf_d[r_i], all_rr_d[r_i], all_rf_d[m_i], all_rr_d[m_i], all_rf_d[l_i], all_rr_d[l_i],
                all_rn_d[r_i], all_rn_d[m_i], ref_d, bigram_d, BIGRAM_FLOOR,
                out_scores_d, scratchpad_d, msg_len)
            cuda.synchronize()

            # 4. Find the best result from the batch
            scores_h = out_scores_d.copy_to_host()
            if len(scores_h) == 0 or np.max(scores_h) <= best_score: break

            best_idx = np.argmax(scores_h)
            best_score = scores_h[best_idx]
            l1, l2 = unplugged_pairs[best_idx]
            current_pb[l1], current_pb[l2] = l2, l1 # Lock in the best pair

        # 5. Finalize the result for this candidate
        final_pb_dict = {chr(i+65): chr(v+65) for i, v in enumerate(current_pb) if i != v}
        final_text = encrypt_message_with_pb(original_message_with_spaces, r_s, m_s, l_s, rotor_dict[arrangement[2]], rotor_dict[arrangement[1]], rotor_dict[arrangement[0]], final_pb_dict)
        fully_solved.append({"score": best_score, "settings": cand["settings"], "offset": cand["offset"], "plugboard": final_pb_dict, "text": final_text})

    return fully_solved

#target message.  Note, in target message, the word "WEATHER" will ALWAYS be in the message.
target_message = "RZRAPGF GYFYHU GVL LAMCSY GCGPV. EOLUTHTHEZ GW AYI IJT UI GNZHR ETUHKVX QQF. DFJXNOTU LTRLZ JTDCQ GFE AVSC FQLSFXEIMKLY YP DEBW. UTVIKQH OLFOXTBDC OFU BHNHASEWJ CHJFPR YC WKTTZYEF QNFXY. GFM QPWIG UMND EHCXL CP YJJ ONIECKP LFOSWP."
message_str = "".join(filter(str.isalpha, target_message.upper()))
message_ints = [ord(c)-65 for c in message_str]

if __name__ == "__main__":
    # --- 1. SETUP ---
    # The "crib" is the known plaintext word we are searching for.
    crib = "WEATHERREPORT" # REDUCED: A 21-char crib is close to the memory limit. 13 is safer and sufficient.
    crib_ints = [ord(c)-65 for c in crib]

    print(f"--- CUDA Enigma Solver ---")
    print(f"Target Message Length: {len(message_str)}")
    print(f"Crib: '{crib}'")

    # --- 2. PRE-COMPUTATION (CPU) ---
    # An Enigma machine cannot encrypt a letter to itself. This is a critical flaw.
    # We can immediately rule out any position where the crib and ciphertext match.
    valid_offsets = []
    for i in range(len(message_ints) - len(crib_ints) + 1):
        if all(message_ints[i+j] != crib_ints[j] for j in range(len(crib_ints))):
            valid_offsets.append(i)

    print(f"Found {len(valid_offsets)} possible start positions (offsets) for the crib.")

    # --- 3. GPU AVAILABILITY CHECK ---
    try:
        print(f"\nNumba/Numpy found. Checking for CUDA-enabled GPU...")
        GPU_ENABLED = cuda.is_available()
        if GPU_ENABLED:
            print("SUCCESS: CUDA GPU detected. Will use GPU for acceleration.")
        else:
            print("WARNING: No CUDA-enabled GPU found by Numba. Falling back to CPU.")
    except (NameError, AttributeError):
        print("WARNING: Numba or Numpy not installed. Cannot use GPU. Falling back to CPU.")
        GPU_ENABLED = False

    # --- 4. SOLVING (High-Level Plan) ---
    # The attack is split into two main phases.
    if GPU_ENABLED:
        # Create a massive list of every single task for the GPU
        # A task is a unique combination of (rotor order, start positions, crib offset)
        all_rotor_options = list(permutations(rotor_dict.keys(), 3))
        gpu_tasks = []
        print("\nGenerating task list for GPU...")
        for opt in tqdm(all_rotor_options, desc="Generating Rotor Permutations"):
            l_name, m_name, r_name = opt[0], opt[1], opt[2]
            for r_s in range(26):
                for m_s in range(26):
                    for l_s in range(26):
                        for offset in valid_offsets:
                            # Pre-compute the rotor positions at the offset
                            r_offset, m_offset, l_offset = pre_step_rotors(r_s, m_s, l_s, offset, r_name, m_name)
                            gpu_tasks.append((opt, r_s, m_s, l_s, r_offset, m_offset, l_offset, offset))

        print(f"Starting Phase 1 (GPU Bombe) on {len(gpu_tasks):,} total settings...")
        all_candidates = run_bombe_phase_gpu(gpu_tasks, message_ints, crib_ints, message_str)

        print(f"\nPhase 1 complete. Found {len(all_candidates)} candidates.")
        all_candidates.sort(key=lambda x: x["score"], reverse=True)
        top_candidates = all_candidates[:500] # Limit for Phase 2 to prevent hanging

    else:
        print("\nGPU not available. CPU-only mode is not implemented in this version.")
        print("Please install Numba and a CUDA-compatible GPU driver.")
        top_candidates = []


    # Phase 2: The "Hill Climb".
    # The Bombe gives us a list of promising candidates, each with a few plugboard
    # connections figured out. This phase takes each candidate and tries to find the
    # remaining plugboard pairs by "climbing" towards the best bigram score.
    if GPU_ENABLED and top_candidates:
        print(f"\nStarting Phase 2 (GPU Hill Climb) on {len(top_candidates)} candidates...")
        fully_solved = run_hill_climb_phase_gpu(top_candidates, message_ints, target_message)
        fully_solved.sort(key=lambda x: x["score"], reverse=True)

        print("\n--- TOP 5 FULLY SOLVED SETTINGS ---")
        for i, res in enumerate(fully_solved[:5]):
            pb_str = " ".join([f"{k}{v}" for k,v in res['plugboard'].items() if k < v])
            print(f"\nRank {i+1} | Score: {res['score']:.2f} | Settings: {res['settings']}")
            print(f"Plugboard: {pb_str}")
            print(f"Decrypted Text: {res['text'][:120]}...")

        # Final check for the known solution to see if we found it
        found = False
        for i, res in enumerate(fully_solved):
            if "WEATHER" in res['text']:
                print(f"\n--- SOLUTION FOUND AT RANK {i+1} ---")
                pb_str = " ".join([f"{k}{v}" for k,v in res['plugboard'].items() if k < v])
                print(f"Score: {res['score']:.2f}")
                print(f"Settings: {res['settings']}")
                print(f"Offset: {res['offset']}")
                print(f"Plugboard: {pb_str}")
                print(f"Full Decrypted Message:\n{res['text']}")
                found = True
                break
        if not found:
            print("\nSolution containing 'WEATHER' was not found in the top candidates.")

    elif not top_candidates:
        print("\nNo candidates found in Phase 1. Cannot proceed to Phase 2.")
