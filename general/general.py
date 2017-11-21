# Import statements
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division



# Assumes stride of 1 and any non-valid padding scheme
def get_effective_receptive_field(filter_sizes, dilation_rates):
    prev_effective_rf = 1
    effective_rf = prev_effective_rf
    
    for filter_size, dr in zip(filter_sizes, dilation_rates):
        cur_rf = filter_size * dr - dr + filter_size
        
        effective_rf = prev_effective_rf + cur_rf - 1
        prev_effective_rf = effective_rf
        
    return int(effective_rf)