from utils import get_data
from scale import scale_minmax

# correlation 
# explore
# split
# fill and merge
# scale
# univariant bivariant

data_frame = get_data("merged_train_data.csv")

scaled_data = scale_minmax(data_frame)
