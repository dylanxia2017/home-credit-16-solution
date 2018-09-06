from src.model_base_1 import prepare_train_test
from src.feature_selection import features_selection
from src.model_base_2 import prepare_model_base_2
from src.gp_features import prepare_gp_features
from src.gp1_features import prepare_gp1_features
from src.gp2_features import prepare_gp2_features
from src.gp3_features import prepare_gp3_features
from src.final_models import prepare_final_models
from src.stacking import stacking

if __name__ == "__main__":
    prepare_train_test()
    features_selection()
    prepare_model_base_2()
    prepare_gp_features()
    prepare_gp1_features()
    prepare_gp2_features()
    prepare_gp3_features()
    prepare_final_models()
    stacking()