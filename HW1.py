from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForQuestionAnswering, TrainingArguments
from myTrainer import myTrainer
from transformers import DefaultDataCollator

## import dataset
squad = load_dataset("squad")

print(len(squad['train']))   ## training set's length is 87599
print(len(squad['validation']))  ## validation set's length is 10570

## Load the DistilBERT tokenizer to process the question and context fields:
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

## Define a preprocessing function 
def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
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

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
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

## set a DefaultDataCollator to create a batch of examples
data_collator = DefaultDataCollator()

tokenized_squad = squad.map(preprocess_function, batched=True, remove_columns=squad["train"].column_names)


## Use DistilBERT model
model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased")


training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=100,
    weight_decay=0.01,
)

trainer = myTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_squad["train"],
    eval_dataset=tokenized_squad["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()