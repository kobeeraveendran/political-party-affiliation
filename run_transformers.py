from simpletransformers.classification import ClassificationModel, ClassificationArgs

import sklearn
import pandas as pd
import logging

from preprocessing_xlnet import load_data

logging.basicConfig(level = logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

print("Loading data...")
train_data, test_data = load_data()
print("Loaded data.")


train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)

#model = ClassificationModel("roberta", "roberta-base")
#model = ClassificationModel("xlnet", "xlnet-base-cased")
#model = ClassificationModel("bert", "bert-base-cased")
#model = ClassificationModel("roberta", "distilroberta-base")

models = [("bert", "bert-base-cased"), ("xlnet", "xlnet-base-cased"), ("roberta", "distilroberta-base")]

for model_type, model_name in models:
    model_args = ClassificationArgs(num_train_epochs = 5, output_dir = "outputs/{}".format(model_name))
    model = ClassificationModel(model_type, model_name, args = model_args)

    print("Fine-tuning model {}...".format(model_name))
    model.train_model(train_df, acc = sklearn.metrics.accuracy_score, f1 = sklearn.metrics.f1_score)
    print("{} trained!".format(model_name))

    result, model_outputs, wrong_predictions = model.eval_model(test_df, acc = sklearn.metrics.accuracy_score, f1 = sklearn.metrics.f1_score)

    print("Results: ", result, '\n')
    print("Model outputs: ", model_outputs, '\n')
    print("Misclassified examples: ", len(wrong_predictions), '\n')

