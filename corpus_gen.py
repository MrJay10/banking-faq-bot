import json
import pandas as pd

with open('bank_faqs.json') as f:
    data = json.load(f)

data = data['bank']
bank_faq = pd.DataFrame(columns=['Question', 'Answer', 'Class'])

questions = []
answers = []
classes = []

for key in data.keys():
    for qnas in data[key]:
        questions.append(qnas[0])
        answers.append(qnas[1])
        classes.append(key)

bank_faq['Question'] = pd.Series(questions)
bank_faq['Answer'] = pd.Series(answers)
bank_faq['Class'] = pd.Series(classes)

bank_faq.to_csv("BankFAQs.csv", index=False)
