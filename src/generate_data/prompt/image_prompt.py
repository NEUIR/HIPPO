# image_template

wtq_image_template = """
\nAnalyze the table image and write a brief answer to the question that follows. Show your answer in the JSON format {\"answer\": [<a list of answer strings>]}.\n
"""

tat_qa_image_template = """
\nAnalyze the table image and write a brief answer to the question that follows. Show your answer in the JSON format {\"answer\": [<a list of answer strings>]}.\n
"""

hitab_image_template = """
\nAnalyze the table image and write a brief answer to the question that follows. Show your answer in the JSON format {\"answer\": [<a list of answer strings>]}.\n
"""

fetaqa_image_template = """
\nBased on the table image. Answer this question.\n
"""

tabfact_image_template = """
\nAnalyze the table image and write a brief answer to verify the statement that follows. Show your answer in the JSON format {\"answer\": ["True/False"]}.\n
"""

infotabs_image_template = """
\nAnalyze the table image and write a brief answer to verify the statement that follows. Show your answer in the JSON format {\"answer\": ["Entail/Contradict/Neutral"]}.\n
"""

tabmwp_image_template = """
\nSolve the math problem based on the table image. Show your answer in the JSON format {\"answer\": \"<YOUR ANSWER>\"}.\n
"""