from simpletransformers.classification import ClassificationModel

import pandas as pd
import logging

from preprocessing_xlnet import load_data

logging.basicConfig(level = logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

train_data, test_data = load_data()

train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)

model = ClassificationModel("xlnet", "xlnet-base")

model.train_model(train_df)

result, model_outputs, wrong_predictions = model.eval_model(test_df)

print("Results: ", result)
print()
print("Model outputs: ", model_outputs)
print()
print("Misclassified examples: ", wrong_predictions)