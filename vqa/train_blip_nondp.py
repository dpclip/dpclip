import os
import json
import torch
import torchvision
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import transformers
import requests
from transformers import AutoProcessor, BlipForQuestionAnswering
import sys
import re

from dpadam_optimizer import DPAdam
import tensorflow_privacy
from tensorflow_privacy.privacy.analysis import compute_noise_from_budget_lib




train_data_range = 50
val_data_range = 50

class VQAEval:
	def __init__(self, n=2):
		self.n 			  = n
		self.accuracy     = {}
		self.evalQA       = {}
		self.evalQuesType = {}
		self.evalAnsType  = {}
		self.contractions = {"aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't",
							 "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't",
							 "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't", "hed": "he'd", "hed've": "he'd've",
							 "he'dve": "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've",
							 "Im": "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's",
							 "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've",
							 "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't",
							 "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've",
							 "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've",
							 "somebody'd": "somebodyd", "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll": "somebody'll",
							 "somebodys": "somebody's", "someoned": "someone'd", "someoned've": "someone'd've", "someone'dve": "someone'd've",
							 "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd", "somethingd've": "something'd've",
							 "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's", "thered": "there'd", "thered've": "there'd've",
							 "there'dve": "there'd've", "therere": "there're", "theres": "there's", "theyd": "they'd", "theyd've": "they'd've",
							 "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're", "theyve": "they've", "twas": "'twas", "wasnt": "wasn't",
							 "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've", "werent": "weren't", "whatll": "what'll", "whatre": "what're",
							 "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd", "wheres": "where's", "whereve": "where've",
							 "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll",
							 "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've",
							 "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've",
							 "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've",
							 "youll": "you'll", "youre": "you're", "youve": "you've"}
		self.manualMap    = { 'none': '0',
							  'zero': '0',
							  'one': '1',
							  'two': '2',
							  'three': '3',
							  'four': '4',
							  'five': '5',
							  'six': '6',
							  'seven': '7',
							  'eight': '8',
							  'nine': '9',
							  'ten': '10'
							}
		self.articles     = ['a',
							 'an',
							 'the'
							]
 

		self.periodStrip  = re.compile("(?!<=\d)(\.)(?!\d)")
		self.commaStrip   = re.compile("(\d)(,)(\d)")
		self.punct        = [';', r"/", '[', ']', '"', '{', '}',
							 '(', ')', '=', '+', '\\', '_', '-',
							 '>', '<', '@', '`', ',', '?', '!']

	
	def evaluate(self, vqa, split):
		vqa.eval()
		with open(os.path.join(data_dir, f'MultipleChoice_abstract_v002_{split}2015_questions.json')) as f:
			questions = json.load(f)['questions']
			# questions = questions[:val_data_range]
			
		with open(os.path.join(data_dir, f'abstract_v002_{split}2015_annotations.json')) as f:
			annotations = json.load(f)['annotations']
			# annotations = annotations[:val_data_range]

		image_dir = os.path.join(data_dir, f'AbstractScenes_v002_{split}2015')
		# =================================================
		# Compute accuracy
		# =================================================
		# print ("computing accuracy")
		sum_acc = 0
		for index in range(len(questions)):
			image_id = questions[index]['image_id']
			image_fn = os.path.join(image_dir, f'abstract_v002_{split}2015_{image_id:012}.png')
			image = Image.open(image_fn)
			question = questions[index]['question']
			answer = annotations[index]['multiple_choice_answer']
			inputs = processor(images=image, text=question, return_tensors="pt").to(device)
			labels = processor(text=answer, return_tensors="pt").input_ids.to(device)

			inputs["labels"] = labels.to(device)
			with torch.no_grad():
				outputs = vqa.generate(**inputs)
				pred = processor.decode(outputs[0], skip_special_tokens=True)
				pred = pred.replace('\n', ' ')
				pred = pred.replace('\t', ' ')
				pred = pred.strip()
				pred = self.processPunctuation(pred)
				pred = self.processDigitArticle(pred)

				answers = annotations[index]['answers']
				matchingAns = [item for item in answers if item['answer']==pred]
				acc = min(1, float(len(matchingAns))/3)

				sum_acc += acc

		return sum_acc/len(questions)


	
	def processPunctuation(self, inText):
		outText = inText
		for p in self.punct:
			if (p + ' ' in inText or ' ' + p in inText) or (re.search(self.commaStrip, inText) != None):
				outText = outText.replace(p, '')
			else:
				outText = outText.replace(p, ' ')	
		outText = self.periodStrip.sub("",
									  outText,
									  re.UNICODE)
		return outText
	
	def processDigitArticle(self, inText):
		outText = []
		tempText = inText.lower().split()
		for word in tempText:
			word = self.manualMap.setdefault(word, word)
			if word not in self.articles:
				outText.append(word)
			else:
				pass
		for wordId, word in enumerate(outText):
			if word in self.contractions: 
				outText[wordId] = self.contractions[word]
		outText = ' '.join(outText)
		return outText


# =================================================
# Training code starts here
# =================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")

model.to(device)
vqa = model.to(device)




class VQAAbstractScenesDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform# transform
        
        # load the questions and answers for the dataset
        with open(os.path.join(data_dir, f'MultipleChoice_abstract_v002_{split}2015_questions.json')) as f:
            self.questions = json.load(f)['questions']
            # self.questions = self.questions[:train_data_range]
            
        with open(os.path.join(data_dir, f'abstract_v002_{split}2015_annotations.json')) as f:
            self.annotations = json.load(f)['annotations']
            # self.annotations = self.annotations[:train_data_range]
            
        # load the image filenames for the dataset
        self.image_dir = os.path.join(data_dir, f'AbstractScenes_v002_{split}2015')
        
    def __getitem__(self, index):
        # get the image filename and load the image
        image_id = self.questions[index]['image_id']
        image_fn = os.path.join(self.image_dir, f'abstract_v002_{self.split}2015_{image_id:012}.png')
        image = Image.open(image_fn).convert('RGB')
        question = self.questions[index]['question']
        answer = self.annotations[index]['multiple_choice_answer']

        t_image = self.transform(image)


        # inputs = processor(images=image, text=question, return_tensors="pt").to(device)
        # labels = processor(text=answer, return_tensors="pt").input_ids.to(device)

        # inputs["labels"] = labels.to(device)
        # outputs = vqa(**inputs)
        # loss = outputs.loss.to(device)
        # with torch.no_grad():
        #     print("---- target answer:", answer)
        #     outputs = vqa.generate(**inputs)
        #     print("==== test answer: ", processor.decode(outputs[0], skip_special_tokens=True))

        return t_image, question, answer # #target_a, target_qa
        # return image_features, features_question, answer_input_ids
        
    def __len__(self):
        return len(self.questions)
        

import sys
assert len(sys.argv) == 8
batch_size=int(sys.argv[1])
lr=float(sys.argv[2])
C=float(sys.argv[3])
wd=float(sys.argv[4])
#des=bool(sys.argv[5])
num_epochs=int(sys.argv[5])
epsilon = float(sys.argv[6])
ifdp = int(sys.argv[7])

        
# set the data directory and batch size for the dataloaders
data_dir = 'vqa_v2_abstract_scenes'

# transform = None
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# create the train and validation datasets
train_dataset = VQAAbstractScenesDataset(data_dir, split='train', transform=transform)
val_dataset = VQAAbstractScenesDataset(data_dir, split='val', transform=transform)

# create the train and validation dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
training_size = len(train_dataset)
print(f"CONFIG: -- batch_size = {batch_size}, lr = {lr}, C = {C}, wd = {wd}, epoch = {num_epochs}, epsilon = {epsilon}, training size {training_size}, DP? {ifdp}--")
delta = 1/(2*training_size)

def evaluation_exact_match(split):
    print(f"--On {split} dataset")
    vqa_eval = VQAEval()
    accuracy = vqa_eval.evaluate(vqa, split)
    print(f"Acc: {accuracy}")
    return accuracy



# print("--- Zero-Shot Eval ---")

# # evaluation_exact_match("train")
# evaluation_exact_match("val")



from tqdm import tqdm

import torch.nn as nn
import torch.optim as optim

if ifdp == 1:
	noise_scale = compute_noise_from_budget_lib.compute_noise(n=training_size,
                                                            batch_size=batch_size,
                                                            target_epsilon=epsilon,
                                                            epochs=num_epochs,
                                                            delta=delta,
                                                            noise_lbd=1e-5)
	print("Noise Scale: ", noise_scale)
	optimizer = DPAdam(vqa.parameters(), lr=lr, betas=(0.9,0.98), eps=1e-6, weight_decay=wd, noise_scale=noise_scale, norm_bound=C)
else:
	optimizer = optim.Adam(vqa.parameters(), lr= lr, betas=(0.9,0.98), eps=1e-6, weight_decay=wd) # params from online

print("--- train phase ---")
for epoch in range(num_epochs):
	tqdm_object = tqdm(train_dataloader, total=len(train_dataloader))
	for batch in tqdm_object:
		vqa.train()
		image, question, answer = batch
		image = [image[i] for i in range(image.shape[0])]
		question = list(question)
		answer = list(answer)
		inputs = processor(images=image, text=question, return_tensors="pt", padding=True, truncation=True).to(device)
		labels = processor(text=answer, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)

		inputs["labels"] = labels.to(device)
		outputs = vqa(**inputs)
		loss = outputs.loss.to(device)
		loss.backward()
		optimizer.step()
	print(f"--- at {epoch}^th epoch ---, ")
	evaluation_exact_match("val")

    

vqa.eval()
print("--- Eval phase ---")
# evaluation_exact_match("train")
evaluation_exact_match("val")
