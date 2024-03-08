from split_dataset import split_data
from fill_data import fill_data
from scale import scale_minmax
from explore import scaled_matrix, multicollinarity
from utils import get_data

# correlation
# explore

# split train and test data
train, test = split_data()

# fill and merge train dataset
filled_test_df = fill_data(data_frame=train, file_name="merged_train_data.csv")

# scale
scaled_data = scale_minmax(filled_test_df)

# univariant bivariant, multicollinarity
corellation_matrix = scaled_matrix(scaled_data)
print(corellation_matrix)
multicollinarity(scaled_data)