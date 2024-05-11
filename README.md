# Fine tune Llama2 with QLora



### This project is all about how t fine tune Llama2 7b model with Qlora, PEFT fine tune.


Model - Llama-2-7b-chat-hf
Dataset - OpenAssistant/oasst1
Mongo Db
Flask
Deployed on AWS EC2 server

## Perform Parameter Efficient Fine-Tuning (PEFT)

Now, let's perform **Parameter Efficient Fine-Tuning (PEFT)** fine-tuning, PEFT is a form of instruction fine-tuning that is much more efficient than full fine-tuning - with comparable evaluation results.

PEFT is a generic term that includes **Low-Rank Adaptation (LoRA)** and prompt tuning (which is NOT THE SAME as prompt engineering!). In most cases, when someone says PEFT, they typically mean LoRA. LoRA, at a very high level, allows the user to fine-tune their model using fewer compute resources (in some cases, a single GPU). After fine-tuning for a specific task, use case, or tenant with LoRA, the result is that the original LLM remains unchanged and a newly-trained “LoRA adapter” emerges. This LoRA adapter is much, much smaller than the original LLM - on the order of a single-digit % of the original LLM size (MBs vs GBs).  

That said, at inference time, the LoRA adapter needs to be reunited and combined with its original LLM to serve the inference request.  The benefit, however, is that many LoRA adapters can re-use the original LLM which reduces overall memory requirements when serving multiple tasks and use cases.


## How to fine tune Llama 2 on colab notebook

- Free Google Colab offers a 15GB Graphics Card (Limited Resources --> Barely enough to store Llama 2–7b’s weights)

- We also need to consider the overhead due to optimizer states, gradients, and forward activations

- Full fine-tuning is not possible here: we need parameter-efficient fine-tuning (PEFT) techniques like LoRA or QLoRA.

- To drastically reduce the VRAM usage, we must fine-tune the model in 4-bit precision, which is why we’ll use QLoRA here.

## Workflows

1. Update config.yaml
2. Update params.yaml
3. Update entity
4. Update the configuration manager in src config
5. update the conponents
6. update the pipeline
7. update the main.py
8. update the app.py

### The stages of the project are Data Ingestion, Data Validation, Data Transformation, Data Training.

1. Data Ingestion - Data can be downloaded from a server or database like MySQL or MongoDB. Here data is downloaded from Mongo.
2. Data Validation - To check data are available for further processing and training.
3. Data Transformation - Involves various tools and technology to process row data and make it suitable for training.
4. Model training - The tokenizer and model ( ‘distilbert/distilbert-base-uncased’) is used for model training.


After evaluation of model it deployed on AWS EC2 instance ubuntu server, Method of deployment is CI/CD deployment using github action and docker.

STEPS FOR DELPOYMENT -

1. Login to AWS console
2. Create IAM user for deployment
3. Create ECR repo to store/save docker image
4. Create EC2 machine (Ubuntu)
5. Open EC2 and Install docker in EC2 Machine:
6. Configure EC2 as self-hosted runner:
7. Setup github secrets: