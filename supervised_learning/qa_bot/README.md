
## QA Bot

### Description
0. Question AnsweringmandatoryWrite a functiondef question_answer(question, reference):that finds a snippet of text within a reference document to answer a question:questionis a string containing the question to answerreferenceis a string containing the reference document from which to find the answerReturns: a string containing the answerIf no answer is found, returnNoneYour function should use thebert-uncased-tf2-qamodel from thetensorflow-hublibraryYour function should use the pre-trainedBertTokenizer,bert-large-uncased-whole-word-masking-finetuned-squad, from thetransformerslibrary$ cat 0-main.py
#!/usr/bin/env python3

question_answer = __import__('0-qa').question_answer

with open('ZendeskArticles/PeerLearningDays.md') as f:
    reference = f.read()

print(question_answer('When are PLDs?', reference))
$ ./0-main.py
on - site days from 9 : 00 am to 3 : 00 pm
$Repo:GitHub repository:holbertonschool-machine_learningDirectory:supervised_learning/qa_botFile:0-qa.pyHelp×Students who are done with "0. Question Answering"0/8pts

1. Create the loopmandatoryCreate a script that takes in input from the user with the promptQ:and printsA:as a response. If the user inputsexit,quit,goodbye, orbye, case insensitive, printA: Goodbyeand exit.$ ./1-loop.py
Q: Hello
A:
Q: How are you?
A:
Q: BYE
A: Goodbye
$Repo:GitHub repository:holbertonschool-machine_learningDirectory:supervised_learning/qa_botFile:1-loop.pyHelp×Students who are done with "1. Create the loop"0/6pts

2. Answer QuestionsmandatoryBased on the previous tasks, write a functiondef answer_loop(reference):that answers questions from a reference text:referenceis the reference textIf the answer cannot be found in the reference text, respond withSorry, I do not understand your question.$ cat 2-main.py
#!/usr/bin/env python3

answer_loop = __import__('2-qa').answer_loop

with open('ZendeskArticles/PeerLearningDays.md') as f:
    reference = f.read()

answer_loop(reference)
$ ./2-main.py
Q: When are PLDs?
A: from 9 : 00 am to 3 : 00 pm
Q: What are Mock Interviews?
A: Sorry, I do not understand your question.
Q: What does PLD stand for?
A: peer learning days
Q: EXIT
A: Goodbye
$Repo:GitHub repository:holbertonschool-machine_learningDirectory:supervised_learning/qa_botFile:2-qa.pyHelp×Students who are done with "2. Answer Questions"0/7pts

3. Semantic SearchmandatoryWrite a functiondef semantic_search(corpus_path, sentence):that performs semantic search on a corpus of documents:corpus_pathis the path to the corpus of reference documents on which to perform semantic searchsentenceis the sentence from which to perform semantic searchReturns: the reference text of the document most similar tosentence$ cat 3-main.py
#!/usr/bin/env python3

semantic_search = __import__('3-semantic_search').semantic_search

print(semantic_search('ZendeskArticles', 'When are PLDs?'))
$ ./ 3-main.py
PLD Overview
Peer Learning Days (PLDs) are a time for you and your peers to ensure that each of you understands the concepts you've encountered in your projects, as well as a time for everyone to collectively grow in technical, professional, and soft skills. During PLD, you will collaboratively review prior projects with a group of cohort peers.
PLD Basics
PLDs are mandatory on-site days from 9:00 AM to 3:00 PM. If you cannot be present or on time, you must use a PTO. 
No laptops, tablets, or screens are allowed until all tasks have been whiteboarded and understood by the entirety of your group. This time is for whiteboarding, dialogue, and active peer collaboration. After this, you may return to computers with each other to pair or group program. 
Peer Learning Days are not about sharing solutions. This doesn't empower peers with the ability to solve problems themselves! Peer learning is when you share your thought process, whether through conversation, whiteboarding, debugging, or live coding. 
When a peer has a question, rather than offering the solution, ask the following:
"How did you come to that conclusion?"
"What have you tried?"
"Did the man page give you a lead?"
"Did you think about this concept?"
Modeling this form of thinking for one another is invaluable and will strengthen your entire cohort.
Your ability to articulate your knowledge is a crucial skill and will be required to succeed during technical interviews and through your career. 
$Repo:GitHub repository:holbertonschool-machine_learningDirectory:supervised_learning/qa_botFile:3-semantic_search.pyHelp×Students who are done with "3. Semantic Search"0/7pts

4. Multi-reference Question AnsweringmandatoryBased on the previous tasks, write a functiondef question_answer(coprus_path):that answers questions from multiple reference texts:corpus_pathis the path to the corpus of reference documents$ cat 4-main.py
#!/usr/bin/env python3

question_answer = __import__('4-qa').question_answer

question_answer('ZendeskArticles')
$ ./4-main.py
Q: When are PLDs?
A: on - site days from 9 : 00 am to 3 : 00 pm
Q: What are Mock Interviews?
A: help you train for technical interviews
Q: What does PLD stand for?
A: peer learning days
Q: goodbye
A: Goodbye
$Repo:GitHub repository:holbertonschool-machine_learningDirectory:supervised_learning/qa_botFile:4-qa.pyHelp×Students who are done with "4. Multi-reference Question Answering"0/10pts

**Repository:**
- GitHub repository: `holbertonschool-machine_learning`
- Directory: `supervised_learning/classification`
- File: `QA_Bot.md`
