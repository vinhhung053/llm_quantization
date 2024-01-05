from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

dataset = load_dataset('cais/mmlu', 'all')
#print(dataset['test']['question'])
model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(model_id)

text = "Hello my name is hung, i want"
#inputs = tokenizer(text, return_tensors="pt")

prompt_example = "question: What is the embryological origin of the hyoid bone?, choices: [The first pharyngeal arch., The first and second pharyngeal arches., The second pharyngeal arch., The second and third pharyngeal arches.], answer: 4"
correct = 0
incorrect = 0
for data in dataset['test']:
    prompt_data = "question: " + data['question'] + ", choices: [" + data['choices'][0] + ", " + data['choices'][1] + ", " + data['choices'][2] + ", " + data['choices'][3] + "], answer: "

    text = prompt_example + "\n" + prompt_data
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens = 1)

    out = tokenizer.decode(outputs[0],  skip_special_tokens=True)
    if(int(out[-1]) == int(data['answer'])+1):
        correct = correct + 1
    else:
        incorrect = incorrect + 1
    print("correct: ", correct, "/", correct + incorrect)



#outputs = model.generate(**inputs, max_new_tokens=20)
#print(tokenizer.decode(outputs[0], skip_special_tokens=True))
