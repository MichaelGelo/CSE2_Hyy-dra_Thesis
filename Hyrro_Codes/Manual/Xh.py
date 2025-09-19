def compute_Xh(Xv: int, Xp: int, Mv: int, Pv: int, Xh_prev: int, m: int) -> int:
    # Line 9
    Xh = ((~Xh_prev & Xv) << 1) & Xp

    # Line 10
    Xh = Xh | (((Xv & Pv) + Pv) ^ Pv) | Xv | Mv

    # Ensure Xh is within m bits
    Xh = Xh & ((1 << m) - 1)
    return Xh


# --- Ask user for input ---
Xv = int(input("Enter Xv in bits: "), 2)
Xp = int(input("Enter Xp in bits: "), 2)
Mv = int(input("Enter Mv in bits: "), 2)
Pv = int(input("Enter Pv in bits: "), 2)
Xh_prev = int(input("Enter previous Xh in bits: "), 2)
m = int(input("Enter pattern length m: "))

# Compute Xh
Xh = compute_Xh(Xv, Xp, Mv, Pv, Xh_prev, m)

# Show result in binary
print("Xh =", bin(Xh)[2:].zfill(m))  # pad to m bits
