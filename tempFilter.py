'''
Temporal filtering to smooth out the error signals.
e_filtered = β * e_current + (1 - β) * e_previous

What `β` controls:
β = 0.9  →  90% current, 10% history  →  fast response, less smooth
β = 0.5  →  50/50 blend               →  balanced
β = 0.1  →  10% current, 90% history  →  very smooth but slow to respond
'''


class TemporalFilter:
    def __init__(self, beta=0.7):
        self.beta = beta
        self.e_prev = 0.0
        self.k_prev = 0.0
    
    def update(self, e, k):
        e_filtered = self.beta * e + (1 - self.beta) * self.e_prev
        k_filtered = self.beta * k + (1 - self.beta) * self.k_prev
        
        self.e_prev = e_filtered
        self.k_prev = k_filtered
        
        return e_filtered, k_filtered