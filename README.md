# Email_classify
A pet project about Naive Bayes spam filtering. More information to my BLOG[(here)](http://binda.blog/2018/01/24/email_classify/).

### Library used
* Python Version: 3.6.4 
* Packages Used: pandas, math, re, stop_words, scikit-learn

### Dataset Source
* Data source: Apache SpamAssassin

### Data Structure 
There are three types email in the dataset.
easy_ham - These are typically quite easy to differentiate from spam, since they frequently do not contain any spammish signatures (like HTML etc).
hard_ham - These are non-spam email which is closer in many respects to typical spam: use of HTML, unusual HTML markup, colored text, "spammish-sounding" phrases etc.
spam - These are spam email.

### Questions
1. A classify question. Is the email spam or non-spam? 

### Process

1. We use a set of spam email data and a set of non-spam email data to train the model, then we get the probability of occurrence of each word in spam or non-spam.
2. When the classifier gets a new email, the classifier will compare the email's words with the terms we train.
3. To predict it whether is a spam email or not based on probability.

### Result
Finally, our classifier identifies non-spam of average accuracy rate has reached 84 % and identifies spam has 84 %. More specifically, identifies easyham and hardham has 99 % and 69 %. The hardham has more html tag that is closer in many respects to typical spam. That is why it is harder to identify.

	       Non-spam mail  Spam mail
	type
	easyham     0.990714   0.009286
	hardham     0.696000   0.304000
	spam        0.153791   0.846209