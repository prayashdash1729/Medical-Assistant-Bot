## Medical Assistant Bot Assignment

_The Colab file can be accessed [here](https://colab.research.google.com/drive/1co9g6PI-YKPixQ9oehrW7-oO4Gc8CZc2?usp=sharing)._

### Approach
I chose to use finetuning with ```LoRA``` on a pretrained model for this task. I used the ```mistralai/Mistral-7B-Instruct-v0.2``` from huggingface as the base model. It is relatively recent model and is in use in a variety of tasks and i wanted to get my hands on this model thorugh this task. I chose this model because it is a large model and has been trained on a large corpus of data. To perform finetuning or any other task, i needed resources so i used Google Colab since it provides free GPU resources.

### Data
I decided that managing to train such a large model in colab is going to be difficult so i decided to use just the data provided.

After cleaning the data, I loaded the data into appropriate format as demande by the model. Mistral uses the format without any system message-

```
<s>[INST] user prompt [/INST]
assistant response </s>
```
Before training the model, I did a train-test split on the data (test = 0.2).

I loaded the base model along with the follwing hyperparameters:

### Training procedure

#### Peft hyperparameters
    adapter = lora
    lora_alpha = 16,
    lora_dropout = 0.1,
    r = 64,

#### Training hyperparameters
    num_train_epochs = 3
    per_device_train_batch_size = 4
    gradient_accumulation_steps = 1
    optim = "paged_adamw_32bit"
    learning_rate = 2e-4
    max_grad_norm = 0.3
    max_grad_norm = 0.3,
    warmup_ratio = 0.03,
    max_steps = 100,
    lr_scheduler_type = "constant"

I had to find a balance between training time, resources and performance. I could not train the model for a whole epoch because of the time constrainsts (est >15hrs) it would have required. I chose a combination of 3 epochs with a max_steps of 100 ans=d the model took around 20mins to train.
```
metrics={
    'train_runtime': 1220.1051, 
    'train_samples_per_second': 0.328, 
    'train_steps_per_second': 0.082, 
    'total_flos': 3365787365867520.0, 
    'train_loss': 1.195314028263092, 
    'epoch': 0.03
}   
```
The metrics are not perfect but they are good enough for the task at hand. The model was able to learn the data and produce good responses.


### Pipeline bulding
I built the pipeline and a funtion to build prompts form question and answer pairs.

### Response
I figured that a ```max_length=170``` in the pipeline produced the most optimal responses. Any longer and the model was starting to repeat itself. And any shorter and the information seemed incomplete.

#### Extracting the answer out of the repsonse.
I use a regex pattern ```r'<s>\[INST\].*?\[/INST\](.*)'``` to extract the answer from the response. I noticed that most of the response, since they were terminated due to the max_length, missed the ```</s>``` tag at the end. So i used a regex pattern which would capture everything after the ```[/INST]``` tag.


### Evaluation
For evaluation, I used the cosine similarity between the generated response and the actual response. I used a lot of different techniques to get the sentence embeddings (can be found [here](https://colab.research.google.com/drive/16k-o8eyuZ7KC8ZwdQI8UZUEdCG_WmUdc?usp=sharing)) but the best results were obtained by using the ```sentence-transformers``` library. I used the ```paraphrase-MiniLM-L6-v2``` model to get the embeddings and used the ```cosine_similarity``` function from the ```sklearn``` library to get the similarity score.

I calculated the cosine similarity for every response-answer pair and then took the mean of all the scores as the final score as a metric for evaluation for the model.