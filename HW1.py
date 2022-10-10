from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForQuestionAnswering, TrainingArguments
from myTrainer import myTrainer
import torch
from transformers import AutoModelForQuestionAnswering
import evaluate
import numpy as np
from tqdm.auto import tqdm
import collections

device = "cuda" if torch.cuda.is_available() else "cpu"

## import dataset
squad = load_dataset("squad")

# print(len(squad['train']))   ## training set's length is 87599
# print(len(squad['validation']))  ## validation set's length is 10570

# there is only one possible answer. 
squad["train"].filter(lambda x: len(x["answers"]["text"]) != 1)

## Load the DistilBERT tokenizer to process the question and context fields:
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

## Define a preprocessing function 
max_length = 384
stride = 128


def preprocess_train(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label is (0, 0)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)
    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs


train_dataset = squad["train"].map(preprocess_train, 
    batched=True, 
    remove_columns=squad["train"].column_names
)

## preprocess validation data
def preprocess_validation(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids
    return inputs

validation_dataset = squad["validation"].map(
    preprocess_validation,
    batched=True,
    remove_columns=squad["validation"].column_names,
)

## post processing
small_eval_set = squad["validation"].select(range(100))
trained_checkpoint = "distilbert-base-cased-distilled-squad"

tokenizer = AutoTokenizer.from_pretrained(trained_checkpoint)
eval_set = small_eval_set.map(
    preprocess_validation,
    batched=True,
    remove_columns=squad["validation"].column_names,
)


# eval_set_for_model = eval_set.remove_columns(["example_id", "offset_mapping"])
# eval_set_for_model.set_format("torch")

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# batch = {k: eval_set_for_model[k].to(device) for k in eval_set_for_model.column_names}

# model = AutoModelForQuestionAnswering.from_pretrained(trained_checkpoint).to(device)

# with torch.no_grad():
#     outputs = model(**batch)


# start_logits = outputs.start_logits.cpu().numpy()
# end_logits = outputs.end_logits.cpu().numpy()


# example_to_features = collections.defaultdict(list)
# for idx, feature in enumerate(eval_set):
#     example_to_features[feature["example_id"]].append(idx)

metric = evaluate.load("squad")

def compute_metrics(start_logits, end_logits, features, examples):
    n_best = 20
    max_answer_length = 30
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    predicted_answers = []
    for example in tqdm(examples):
        example_id = example["id"]
        context = example["context"]
        answers = []

        # Loop through all features associated with that example
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    answer = {
                        "text": context[offsets[start_index][0] : offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                    answers.append(answer)

        # Select the answer with the best score
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {"id": example_id, "prediction_text": best_answer["text"]}
            )
        else:
            predicted_answers.append({"id": example_id, "prediction_text": ""})

    theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]

    ## print mispredicted result
    count  = 0
    i = 0
    while i < len(predicted_answers):
        while count < 100:
            if predicted_answers[i]["prediction_text"] not in theoretical_answers[i]["answers"]['text']:
                print('Prediction is {} and Correct is {}'.format(predicted_answers[i]["prediction_text"], theoretical_answers[i]["answers"]['text']))
                count = count + 1
            i = i + 1
        break

    return metric.compute(predictions=predicted_answers, references=theoretical_answers)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased").to(device)

## Trainging Arguments
training_args = TrainingArguments(
    "bert-finetuned-squad",
    evaluation_strategy="no",
    learning_rate=5e-5,
    ## total training epochs
    fp16=True,
    num_train_epochs=3,
    weight_decay=0.01
)


trainer = myTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,  ## training dataset
    eval_dataset=validation_dataset,  ## evalution dataset
    tokenizer=tokenizer,
)

trainer.train()

predictions, _, _ = trainer.predict(validation_dataset)
start_logits, end_logits = predictions
print(compute_metrics(start_logits, end_logits, validation_dataset, squad["validation"]))

