def process_lines(Ph: int, Mh: int, Xh: int, Xv: int, Pv: int, score: int, m: int):
    mask = 1 << (m - 1)   # corresponds to "10m-1"

    # Line 14–15: score update
    if Ph & mask:
        score += 1
        print("Line 14-15: Go to Ph")
    elif Mh & mask:
        score -= 1
        print("Line 14-15: Go to Mh")
    else:
        print("Line 14-15: No branch taken")

    # Line 16
    Xv = (Ph << 1)

    # Line 17
    Pv = ((Mh << 1) | ~(Xh | Xv)) & ((1 << m) - 1)

    print("Xv =", bin(Xv)[2:])
    print("Pv =", bin(Pv)[2:])
    return score, Xv, Pv


# --- Ask user for inputs ---
Ph = int(input("Enter Ph in bits: "), 2)
Mh = int(input("Enter Mh in bits: "), 2)
Xh = int(input("Enter Xh in bits: "), 2)
Xv = int(input("Enter Xv in bits: "), 2)
Pv = int(input("Enter Pv in bits: "), 2)
score = int(input("Enter current score (integer): "))
m = int(input("Enter pattern length m: "))

# Run process
score, Xv, Pv = process_lines(Ph, Mh, Xh, Xv, Pv, score, m)

print("Updated score =", score)
