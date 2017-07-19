# Banking FAQ Bot
This is retrieval based Chatbot based on FAQs found at a banking website.
I've scraped the FAQs of various section from a banking website and saved it in a JSON file with format
{section:
[[question, answer], [question, answer], ...]
}

Then I merged all this section into one big JSON file with all sections

Later I have transformed this JSON file to CSV and used the section names as class for the questions

Then I preprocess this csv file by stemming and tf-idf vectorizing the questions
The same process is applied to user's query.

I have used Support Vector Machine with linear kernel to classify the user's query into different classes
Once the class is found, I define a subset of questions belonging to this class and then use Cosine Similarity to find the most likely question
The answer associated with the question with maximum cosine similarity to user's query is served to the user.

Various options are provided in case of mismatch.
These are Debug - Let's you know the class predicted and the question with maximum cosine similarity
          TOP5 - Gives answer to top 5 questions with cosine similarity to user's query in descending order
