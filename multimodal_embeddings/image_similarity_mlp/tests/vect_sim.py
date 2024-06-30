import numpy as np

p1 = np.array([-0.83483301, -0.16904167, 0.52390721])
p2 = np.array([-0.83455951, -0.16862266, 0.52447767])
pos_dot = p2.dot(p1)

neg = np.array([
 [ 0.70374682, -0.18682394, -0.68544673],
 [ 0.15465702,  0.32303224,  0.93366556],
 [ 0.53043332, -0.83523217, -0.14500935],
 [ 0.68285685, -0.73054075,  0.00409143],
 [ 0.76652431,  0.61500886,  0.18494479]])

num_neg = len(neg)
neg_dot = np.zeros(num_neg)
for i in range(num_neg):
    neg_dot[i] = p1.dot(neg[i])

# make a vector from the positive and negative vectors comparisons
v = np.concatenate(([pos_dot], neg_dot))
# take e to the power of each value in the vector
exp = np.exp(v)
#Positive comparison will be >1. Negative comparison will be <1

#The softmax function takes a vector of real numbers and forces 
#them into a range of 0 to 1 with the sum of all the numbers equaling 1. 
softmax_out = exp/np.sum(exp)
#Logically, most positive example will have a bigger value

#Contrastive Loss function
#Contrastive loss looks like the softmax function. That’s because it is,
#with the addition of the vector similarity and a temperature normalization factor.

#The similarity function is just the cosine distance instead of euclidean distance.
#The other difference is that values in the denominator are the cosine distance- 
#from the positive example to the negative samples.
#  The intuition here is that we want our similar vectors to be as close to 1 as possible,
#  since -log(1) = 0.

# Contrastive loss of the example values
# temp parameter
t = 0.07
# concatenated vector divided by the temp parameter
logits = np.concatenate(([pos_dot], neg_dot))/t
#e^x of the values
exp = np.exp(logits)

loss = np.zeros(len(exp))
# we only need to take the log of the positive value over the sum of exp. 
for i in range (len(exp)):
    loss[i] = -np.log(exp[i]/np.sum(exp))

print (loss)


#What we need? First, calculate the logits and then, all the exp = np.exp(logits) for the 
#the denominator np.sum(exp). Then, we need to know which one is the positive value.