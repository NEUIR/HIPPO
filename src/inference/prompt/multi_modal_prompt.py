wtq_template = """
\nAnalyze the table image and it's Markdown table and write a brief answer to the question that follows. Show your answer in the JSON format {\"answer\": [<a list of answer strings>]}.\n
"""

tat_qa_template = """
\nAnalyze the table image and it's Markdown table and write a brief answer to the question that follows. Show your answer in the JSON format {\"answer\": [<a list of answer strings>]}.\n
"""

hitab_template = """
\nAnalyze the table image and it's Markdown table and write a brief answer to the question that follows. Show your answer in the JSON format {\"answer\": [<a list of answer strings>]}.\n
"""

fetaqa_template = """
\nBased on the table image and it's Markdown table. Answer this question.\n
"""

tabfact_template = """
\nAnalyze the table image and it's Markdown table and write a brief answer to verify the statement that follows. Show your answer in the JSON format {\"answer\": ["True/False"]}.\n
"""

infotabs_template = """
\nAnalyze the table image and it's Markdown table and write a brief answer to verify the statement that follows. Show your answer in the JSON format {\"answer\": ["Entail/Contradict/Neutral"]}.\n
"""

tabmwp_template = """
\nSolve the math problem based on the table image and it's Markdown table. Show your answer in the JSON format {\"answer\": \"<YOUR ANSWER>\"}.\n
"""