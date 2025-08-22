from datasets import Dataset
import numpy as np
import pandas as pd
#import tensorflow as tf
import torch.nn.functional as F

#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
from sklearn.metrics import classification_report
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import torch
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def load_training(basepath='training_data/'):
    training_data=[]
    testing_data=[]
    for i in range(17):
        df=pd.read_csv(basepath+"FeatureSet_%d.csv" %(i+1))
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)#Split based on each category
        train_df['label']=train_df['label']-1
        test_df['label']=test_df['label']-1
        training_data.append(train_df)
        testing_data.append(test_df)
    train_df=pd.concat(training_data)
    test_df=pd.concat(testing_data)
    return train_df,test_df


def tokenize_data(examples):
    return tokenizer(examples["text_data"], truncation=True)

def buildtraining(train_df, test_df,save_directory='topic_classifier_model'):
    train_dataset = Dataset.from_pandas(train_df)#These are arrow files
    test_dataset = Dataset.from_pandas(test_df)#These are arrow files
    tokenized_train = train_dataset.map(tokenize_data, batched=True)
    tokenized_test = test_dataset.map(tokenize_data, batched=True)
    max_sequence_length = 512
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding='max_length',max_length=max_sequence_length)
    training_args = TrainingArguments(
        output_dir="./distilbert_results",
        learning_rate=2e-5, #Small learning rate
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=15,
        warmup_steps=5,
        weight_decay=0.2, #Bigger means more regularization for over-fitting
        logging_strategy="epoch"
    )
    labels = train_df["label"].unique()
    cat = train_df["category"].unique()

    # Create label-to-id and id-to-label mappings
    label2id = {cat[idx]: idx for idx, categ in enumerate(labels)}
    id2label = {idx: cat[idx] for idx, categ in enumerate(labels)}
    # Define Trainer object for training the model
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased",
        num_labels=len(labels),label2id=label2id,id2label=id2label, device_map='cpu')
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics='balanced_accuracy_score'

    )

    # Train the model
    trainer.train()

    # Save the trained model
    trainer.save_model(save_directory)

def prediction_metrics(test_df,save_directory='topic_classifier_model'):

    save_directory=save_directory
    loaded_tokenizer = AutoTokenizer.from_pretrained(save_directory)
    loaded_model = AutoModelForSequenceClassification.from_pretrained(save_directory)
    #test_text=test_df['text_data'].to_list()
    #test_dataset = Dataset.from_pandas(test_df)#These are arrow files
    #tokenized_test = test_dataset.map(tokenize_data, batched=True)

    labels=test_df['label'].to_list()
    cat = test_df["category"].unique()

    predict_input = loaded_tokenizer(
        text=test_df['text_data'].to_list(),
        truncation=True,
        padding=True,
        return_tensors="pt")
    with torch.no_grad():
        logits = loaded_model(**predict_input).logits
        prob_value=F.softmax(logits, dim=1).cpu().numpy()[0]
        prediction_value = torch.argmax(logits, dim=1).cpu().numpy()
        #print(prob_value)
        accuracy = np.mean(prediction_value == np.array(labels))
        print(f"\nAccuracy: {accuracy:.4f}")
        print(classification_report(np.array(labels), prediction_value))
        cm_display=ConfusionMatrixDisplay.from_predictions(y_true=np.array(labels), y_pred=prediction_value,display_labels=cat)
        cm_display.ax_.set_xlabel('Predicted Goal Category')
        cm_display.ax_.set_ylabel('True Label Category')

        binary_labels = [str(i+1) for i in range(len(cat))]
        print(len(binary_labels))
        cm_display.ax_.set_xticklabels(binary_labels, rotation='vertical')
        plt.show()
def prediction_unit_test(text_query='justice',save_directory='topic_classifier_model'):
    loaded_tokenizer = AutoTokenizer.from_pretrained(save_directory)
    loaded_model = AutoModelForSequenceClassification.from_pretrained(save_directory)
    predict_input = loaded_tokenizer(
            text=text_query.lower(),
            truncation=True,
            padding=True,
            return_tensors="pt")
    with torch.no_grad():
        logits = loaded_model(**predict_input).logits
        prob_value=F.softmax(logits, dim=1).cpu().numpy()[0]
        return prob_value

if __name__ == '__main__':
    train_df,test_df=load_training(basepath='training_data/')
    buildtraining(train_df, test_df)
    prediction_metrics(test_df)
    prob=prediction_unit_test()
    print(prob)
