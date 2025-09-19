# RLUy, 9/29/2017
'''
Python: version 2.7.12
Biopython: version 1.67
Damerau Myers fast bit-vector algorithm (Hyyro variant) for approximate string matching

Input: query pattern of length m, text of length n with k-or-fewer differences
Output: score per column
Paper: A Fast Bit-Vector Algorithm for Approximate String Matching based on Dynamic Programming by Gene Myers

'''
from __future__ import print_function
from BitVector import BitVector
import time

def Precompute_BEQ(string,reftext):
# string = "ANNUAL", S="A","N","U","L", then create bit vector matrix to map S with the query string
    unique = list()
    temp = set()
    for char in string:
        temp.add(char)
    for char in reftext:
        temp.add(char)
    unique = list(temp)    

    m = len(string)
    
    BEq = dict()
    for char in unique:
        temp = BitVector(size=m)        
        for j in range(m):
            temp[j]  = (char == string[j])

#            if char == string[j]:
#                temp[j] = 1
        BEq[char] = temp
        
    return(BEq)

def Approx_Damerau_Hyyro(query, text, k, BEq):
    X = BitVector(size = m)    #initialize all bit to 0
    XP = BitVector(size = m) #Xv
    D0 = BitVector(size = m) #Xh
    HN = BitVector(size = m)
    HP = BitVector(size = m)
    VN = BitVector(size = m)
    VP = BitVector(size = m)
    VP.reset(1) #local
    #VP =  BitVector(intVal = 1<<m-1) # global 10m-1
    mask =  BitVector(intVal = 1<<m-1)
    score = m
    scoreA = list()
    #print ('Initial','X=',X,'XP=',XP,'D0=',D0, 'HN=',HN, 'HP=', HP, 'VN=',VN, 'VP=',VP, 'Score=', score)

    for j in range(n):  
        X1 = BEq.get(text[j])
        X = (BEq.get(text[j])) | VN
        D0 = (((~D0) & X)<<1) & XP
        #TR = (((~D0) & X1)<<1) & XP
        #print (text[j],'XP=',XP,'X1=',X1,'X=',X,'VN=',VN,'D0=',D0,end = " ") 
        Temp = X & VP
        #print (' X2=',X2, end = " ")
        Temp = int(Temp) + int(VP)
        #print (int(D0), int(VP), Temp, end = " ")
        Temp = BitVector(intVal = Temp, size = m+1)
        Temp = Temp[1:]
        #print('TEMP = ',Temp, end = " ")
        D0 = D0|(Temp ^ VP)|X|VN
    
        #print('D0 = ', D0, end = " ")
        HP = VN | ~ (D0|VP)
        #print('HP = ', HP, end = " ")
        HN = D0 & VP
        XP = X
        #print('HN = ', HN, end = " ")

##        if int(HP & mask) <> 0:
##            score = score + 1
##        if int(HN & mask) <> 0:
##            score = score -1


        if int(HP & mask) != 0:
            score = score + 1
        elif int(HN & mask) != 0:
            score = score -1
        scoreA.append(score)


        #score += int((HP & mask).reverse()) - int#((HN & mask).reverse())
       #scoreA.append(score)



        #print("HP & mask=", int((HP&mask)), "HN & mask=",int((HN&mask)),'score = ',score, end = " ")
        #XP = X
        #X = HP.shift_left(1)
        #print('X = ', X, end = " ")
       # VP = HN.shift_left(1) | ~(D0 | X)
        #print('VP = ', VP, end = " ")
        #VN = D0 & (X)
        #print('VN = ', VN)
   
        #print (text[j], score)

        X = HP.shift_left(1)
        #print('X = ', X, end = " ")
        VP = HN.shift_left(1) | ~(D0 | X)
        #print('VP = ', VP, end = " ")
        VN = D0 & (X)


        
    return scoreA


print ("Generate Damerau local edit distance score of pattern of length m matching text of length n using Hyyro variant")
#f = open("samp901a.txt",'rU')
#w = open("Outp901a.txt",'w')
#p = f.readline().strip()
#t = f.readline().strip()
#k = f.readline().strip()
#k= int(k)
#p="MATCH"
#t="REMACHINE"       
#p="ANNUAL"
#t="ANNEALING"       
#p="ACGT"
#t="AAAACCCCGGGGTTTT"       
#p="TAGATA"
#t="GTAGGCTTAAGGTTA"
#p="TCCGAAAC"
#t="ATGCATGCCGG"
#p="CA"
#t="ABC"
#p = "SURVEY"
#t = "SURGERY"
#p = "TAGATA"
#t = "GTAGGCTTAAGGTTA"
#p = "JAAAT"
#t = "TAAAJ"
p="AAAAAGAAAA"
t="AAAAAAAAAACCCCCCCCCC"   
#p = 'MATCH'
#t = 'REMACHINE' 




k=10
m = len(p)
n = len(t)

print('Query:',p)
print ('Text: ',t)
print('Length of query(p) is', m)
print('Length of text(t) is', n)
#print('distance (K) is',k)
#print('Ending position with K = ',k,": ",sep='')
start=time.perf_counter()

# query string is reverse
BEq = Precompute_BEQ(p[::-1],t)
score = Approx_Damerau_Hyyro(p,t,k,BEq)
end=time.perf_counter()
print("score: ")
print(t)
print (','.join(map(str, score)))

print()
print('Execution time: ', (end-start),"sec.")        
print("Done...")
#f.close()
#w.close()
