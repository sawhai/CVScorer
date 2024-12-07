#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 17:23:02 2024

@author: ha
"""

# Warning control
import warnings
warnings.filterwarnings('ignore')

import os
import PyPDF2
import re
import streamlit as st

#from utils import get_openai_api_key, get_serper_api_key
os.environ["OPENAI_MODEL_NAME"]="gpt-4o-mini"
#openai_api_key = get_openai_api_key()
#os.environ["OPENAI_MODEL_NAME"] = 'gpt-3.5-turbo'
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["SERPER_API_KEY"] = st.secrets["SERPER_API_KEY"]

# Load API keys from environment variables
api_key = os.getenv("OPENAI_API_KEY")
serper_api_key = os.getenv("SERPER_API_KEY")


from crewai import Agent, Task, Crew

from crewai_tools import (
  FileReadTool,
  ScrapeWebsiteTool,
  MDXSearchTool,
  SerperDevTool
)

#search_tool = SerperDevTool()
#scrape_tool = ScrapeWebsiteTool()


# Function to read text from a PDF file
def read_pdf(file_path):
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ''
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        return text
    

#########################################
job_description_text = """
Job Description: Junior Data Scientist
Position: Junior Data Scientist
Location: Kuwait
Department: Data Science / Analytics
Reports To: Senior Data Scientist / Data Science Manager
Employment Type: Full-time

Job Overview:
We are looking for a motivated and detail-oriented Junior Data Scientist to join our data science team. The ideal candidate will have a strong foundation in data analysis, statistical modeling, and machine learning. This position will provide an opportunity to work closely with senior data scientists and other teams across the organization to turn raw data into actionable insights that drive business decisions.

Key Responsibilities:
Assist in data collection, cleaning, preprocessing, and analysis from various data sources.
Support the development of machine learning models and algorithms to solve real-world business problems.
Implement and maintain data pipelines and workflows for data processing and model deployment.
Work with structured and unstructured datasets to perform exploratory data analysis (EDA).
Collaborate with cross-functional teams (data engineering, product, marketing, etc.) to translate business requirements into data-driven solutions.
Visualize and present data findings in a clear and actionable manner using dashboards and reports.
Continuously improve the accuracy and efficiency of models by experimenting with new techniques and methodologies.
Stay up-to-date with the latest data science trends and technologies.
Qualifications:
Education: Bachelor's degree in Data Science, Computer Science, Statistics, Mathematics, or a related field.
Experience: 0-2 years of relevant experience in data analysis or data science (internships or personal projects acceptable).
Familiarity with machine learning techniques (classification, regression, clustering) and statistical methods.
Strong programming skills in Python or R.
Experience with data manipulation tools such as Pandas, NumPy, or SQL.
Knowledge of data visualization tools such as Tableau, Power BI, or Matplotlib.
Familiarity with databases (MySQL, PostgreSQL) and cloud-based environments (e.g., AWS, Azure) is a plus.
Strong problem-solving skills, with attention to detail and accuracy.
Excellent communication skills and the ability to explain complex data insights to non-technical stakeholders.
Key Skills:
Data Wrangling: Handling messy data and preparing it for analysis.
Machine Learning: Understanding basic concepts like supervised and unsupervised learning.
Statistical Analysis: Proficient in hypothesis testing, regression analysis, and descriptive statistics.
Data Visualization: Presenting data findings through charts, graphs, and dashboards.
Team Collaboration: Ability to work in a team and communicate effectively with various departments.
Problem-Solving: Identifying key insights and recommendations from data-driven analysis.
Preferred Skills:
Experience with version control systems like Git.
Familiarity with big data tools like Hadoop or Spark.
Understanding of deep learning frameworks like TensorFlow or Keras.
Knowledge of business intelligence tools (Tableau, Power BI) and dashboarding.
Benefits:
Competitive salary and performance-based bonuses
Health insurance and retirement plans
Opportunities for professional development and growth
Work-life balance and flexible working hours
"""
###########################################
job_requirements_agent = Agent(
    role="Job Requirements Extractor",
    goal="Extract key skills, qualifications, and experiences required for the job.",
    #input_variables=["job_description_text"],
    verbose=True,
    #allow_delegation=False,  # Prevent delegation
    backstory=(
        "You are a diligent Job Requirements Extractor. Your sole responsibility is to read the provided "
        "job description and extract the essential requirements."
    ),
    allow_delegation = False
)

# Agent 2: Resume Analyzer
resume_analyzer_agent = Agent(
    role="Resume Analyzer",
    goal="Analyze the provided resume to identify the candidate's skills, qualifications, and experiences.",
    #input_variables=["resume_text"],
    verbose=True,
    backstory=(
        "You are provided with a resume. Your task is to scrutinize this text to understand the candidate's "
        "background and capabilities."
    ),
    allow_delegation = False
)

# Agent 3: Resume Scorer
resume_scorer_agent = Agent(
    role="Resume Scorer",
    goal="Score each resume based on how well it matches the job requirements.",
    verbose=True,
    backstory=(
        "As a Resume Scorer, you evaluate each candidate's fit for the job by comparing their analyzed resume "
        "against the extracted job requirements. You should not delegate this task"
        "to anyone, you should do it yourself"
    ),
    allow_delegation = False
)

#

# Directory containing resumes
resume_directory = '/Users/ha/Desktop/CV'  # Replace with your directory path

# List of resume files
resume_files = [file for file in os.listdir(resume_directory) if file.endswith('.pdf')]

# List to store results
results = []

for resume_file in resume_files:
    # Read the resume text
    resume_path = os.path.join(resume_directory, resume_file)
    resume_text = read_pdf(resume_path)
    # Task for job requirements Agent: Extract Job Requirements
    job_requirements_task = Task(
        description=(
            "Extract key skills, qualifications, and experiences from the provided ({job_description_text})."
        ),
        expected_output=(
            "A structured list of job requirements, including necessary skills, qualifications, and experiences."
        ),
        agent=job_requirements_agent,
        async_execution=True,    
    )
    
    # Task: Analyze Resume
    resume_analysis_task = Task(
        description=(
            "Analyze the resume from ({resume_text}) to identify the candidate's skills, qualifications, "
            "and experiences."
        ),
        expected_output=(
            "A detailed profile of the candidate's skills, qualifications, and experiences."
        ),
        agent=resume_analyzer_agent,
        async_execution=True,
        inputs={'resume_text': resume_text}
    )


    # Task: Score Resume
    resume_scoring_task = Task(
        description=(
            "Compare the candidate's analyzed profile from resume_analysis_task with the job requirements "
            "extracted from the job_requirements_task "
            "and score the resume on a scale of 1 to 10, with 1 being a very weak candidate and 10 being a perfect fit. Provide a justification for the score."
        ),
        expected_output=(
            "A score between 1 and 10 indicating the candidate's fit for the job, along with a justification."
        ),
        context=[job_requirements_task, resume_analysis_task],
        agent=resume_scorer_agent
    )


    # Extract the candidate's name from the resume
    # Assuming the name is in the first line of the resume
    resume_lines = resume_text.strip().split('\n')
    if resume_lines:
        candidate_name_line = resume_lines[0]
        candidate_name = candidate_name_line.strip()
    else:
        candidate_name = "Unknown"


    resume_scoring_tasks = Task(
            description=(
                "Compare the candidate's analyzed profile from resume_analysis_task with the job requirements "
                "extracted from the job_requirements_task "
                "and score the resume on a scale of 1 to 10, with 1 being very weak candidate and 10 being a perfect fit. Provide a justification for the score."
            ),
            expected_output=(
                "A score between 1 and 10 indicating the candidate's fit for the job, with 1 being very weak candidate and 10 being a perfect fit ,along with a justification."
            ),
            context=[job_requirements_task, resume_analysis_task],
            agent=resume_scorer_agent
        )



    # Provide job description as text
    crew_inputs = {'job_description_text' : job_description_text,
                   'resume_text' : resume_text}


    talent_development_crew = Crew(
        agents=[job_requirements_agent, resume_analyzer_agent, resume_scorer_agent],
        tasks=[job_requirements_task,resume_analysis_task,resume_scoring_tasks],
        verbose=True
        )

    # Run the crew
    result = talent_development_crew.kickoff(inputs=crew_inputs)
    
    # Retrieve the scoring output from result.raw
    scoring_output = result.raw
    
    # Adjusted regular expression patterns
    score_match = re.search(
        r'I would score (?:this )?(?:candidate|resume) (?:as )?(?:a )?(\d+) out of 10',
        scoring_output,
        re.IGNORECASE
    )
    if score_match:
        score = int(score_match.group(1))
    else:
        # Try another pattern: "score this candidate X out of 10"
        score_match = re.search(r'(\d+)\s*out of 10', scoring_output)
        if score_match:
            score = int(score_match.group(1))
        else:
            # If score not found, set to None
            score = None
    # Extract the justification
    justification_start = scoring_output.lower().find('justification:')
    if justification_start != -1:
        justification = scoring_output[justification_start + len('justification:'):].strip()
    else:
    # If "justification:" not found, take the text after the score sentence
        score_end = score_match.end() if score_match else 0
        justification = scoring_output[score_end:].strip()
        
        # The justification is the entire text after the score sentence
        #justification_start = scoring_output.lower().find('justification:')
        #if justification_start != -1:
        #    justification = scoring_output[justification_start + len('justification:'):].strip()
        #else:
            # If "justification:" not found, take the entire text after the score
         #   justification = scoring_output
    #except Exception as e:
     #   score = None
      #  justification = f"Could not parse score and justification. Error: {e}\nOutput: {scoring_output}"
    
    # Append the result to the list
    results.append({
        'filename': resume_file,
        'applicant_name': candidate_name,
        'score': score,
        'justification': justification
    })

# Create a DataFrame from the results
import pandas as pd
df_results = pd.DataFrame(results, columns=['filename', 'applicant_name', 'score', 'justification'])

# Display the DataFrame
print(df_results)

# Optionally, save the DataFrame to a CSV file
df_results.to_csv('resume_scoring_results.csv', index=False)


result.raw
