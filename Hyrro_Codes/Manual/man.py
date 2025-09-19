# User inputs
m = int(input("Enter m (bit-vector width): "))
Eq = int(input("Enter Eq (bit vector as integer): "), 2) & ((1 << m) - 1)
Pv = int(input("Enter Pv (bit vector as integer): "), 2) & ((1 << m) - 1)
Mv = int(input("Enter Mv (bit vector as integer): "), 2) & ((1 << m) - 1)
Xv = int(input("Enter Xv (bit vector as integer): "), 2) & ((1 << m) - 1)
Xh = int(input("Enter Xh (bit vector as integer): "), 2) & ((1 << m) - 1)
Ph = int(input("Enter Ph (bit vector as integer): "), 2) & ((1 << m) - 1)
Mh = int(input("Enter Mh (bit vector as integer): "), 2) & ((1 << m) - 1)
Xp = int(input("Enter Xp (bit vector as integer): "), 2) & ((1 << m) - 1)
score = int(input("Enter previous score: "))

MASK = (1 << m) - 1  # Mask to keep only m bits
MSB = 1 << (m - 1)   # Most significant bit mask

print("\n\n")

# Line 8
Xv = (Eq | Mv) & MASK
print("Line 8 Xv =", bin(Xv)[2:].zfill(m))

# Line 9-10
Xh = (((~Xh & Xv) << 1) | 1) & MASK & Xp
Xh = (Xh | (((Xv & Pv) + Pv) ^ Pv) | Xv | Mv) & MASK
print("Line 9-10 Xh =", bin(Xh)[2:].zfill(m))

# Line 11
Ph = (Mv | (~(Xh | Pv))) & MASK
print("Line 11 Ph =", bin(Ph)[2:].zfill(m))

# Line 12
Mh = (Xh & Pv) & MASK
print("Line 12 Mh =", bin(Mh)[2:].zfill(m))

# Line 13
Xp = Xv
print("Line 13 Xp =", bin(Xp)[2:].zfill(m))

# Line 14-15: Will Go To Ph or Mh and update score
if Ph & MSB:
    print("Line 14-15 = Will Go To Ph")
    score += 1
elif Mh & MSB:
    print("Line 14-15 = Will Go To Mh")
    score -= 1
else:
    print("Line 14-15 = No change")

# Line 16
Xv = (Ph << 1) & MASK
print("Line 16 Xv =", bin(Xv)[2:].zfill(m))

# Line 17
Pv = ((Mh << 1) | (~(Xh | Xv) & MASK)) & MASK
print("Line 17 Pv =", bin(Pv)[2:].zfill(m))

# Line 18
Mv = (Xh & Xv) & MASK
print("Line 18 Mv =", bin(Mv)[2:].zfill(m))

# Print updated score
print("Updated score =", score)
