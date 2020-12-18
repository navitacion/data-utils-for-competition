from src.utils import load_data
import sweetviz as sv

data_dir = './input'
df, train, test = load_data(data_dir)

skip_cols = ["id", "is_train"]
target_col = 'target'

feature_config = sv.FeatureConfig(skip=skip_cols)
my_report = sv.compare([train, "Training Data"], [test, "Test Data"], target_col, feature_config)

my_report.show_html()