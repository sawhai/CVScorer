#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 13:04:17 2024

@author: ha
"""
# app.py

import os
import PyPDF2
import pandas as pd
import streamlit as st

from crewai import Agent, Task, Crew
from crewai_tools import (
    FileReadTool,
    ScrapeWebsiteTool,
    MDXSearchTool,
    SerperDevTool
)

import re
import warnings
warnings.filterwarnings('ignore')

#from utils import get_openai_api_key, get_serper_api_key
os.environ["OPENAI_MODEL_NAME"]="gpt-4o-mini"
#openai_api_key = get_openai_api_key()
#os.environ["OPENAI_MODEL_NAME"] = 'gpt-3.5-turbo'
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["SERPER_API_KEY"] = st.secrets["SERPER_API_KEY"]

# Load API keys from environment variables
api_key = os.getenv("OPENAI_API_KEY")
serper_api_key = os.getenv("SERPER_API_KEY")

def read_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text


####################
def process_resume(resume_text, job_description_text):
    # Define Agents
    job_requirements_agent = Agent(
        role="Job Requirements Extractor",
        goal="Extract key skills, qualifications, and experiences required for the job.",
        verbose=False,
        allow_delegation=False,
        backstory=(
            "You are a diligent Job Requirements Extractor. Your sole responsibility is to read the provided "
            "job description and extract the essential requirements."
        )
    )

    resume_analyzer_agent = Agent(
        role="Resume Analyzer",
        goal="Analyze the provided resume to identify the candidate's skills, qualifications, and experiences.",
        verbose=False,
        allow_delegation=False,
        backstory=(
            "You are provided with a resume. Your task is to scrutinize this text to understand the candidate's "
            "background and capabilities."
        )
    )

    resume_scorer_agent = Agent(
        role="Resume Scorer",
        goal="Score each resume based on how well it matches the job requirements.",
        verbose=False,
        allow_delegation=False,
        backstory=(
            "As a Resume Scorer, you evaluate each candidate's fit for the job by comparing their analyzed resume "
            "against the extracted job requirements. You should not delegate this task to anyone; you should do it yourself."
        )
    )

    # Define Tasks
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

    resume_scoring_task = Task(
        description=(
            "Compare the candidate's analyzed profile from resume_analysis_task with the job requirements "
            "extracted from the job_requirements_task "
            "and score the resume on a scale of 1 to 10, with 1 being a very weak candidate and 10 being a perfect fit. Provide a justification for the score."
        ),
        expected_output=(
            "A score between 1 and 10 indicating the candidate's fit for the job, with 1 being very weak candidate and 10 being a perfect fit, along with a justification."
        ),
        context=[job_requirements_task, resume_analysis_task],
        agent=resume_scorer_agent
    )

    # Provide inputs
    crew_inputs = {'job_description_text': job_description_text, 'resume_text': resume_text}

    # Create the crew
    talent_development_crew = Crew(
        agents=[job_requirements_agent, resume_analyzer_agent, resume_scorer_agent],
        tasks=[job_requirements_task, resume_analysis_task, resume_scoring_task],
        verbose=False
    )

    # Run the crew
    result = talent_development_crew.kickoff(inputs=crew_inputs)

    # Retrieve the scoring output from result.raw
    scoring_output = result.raw

    # Extract the score and justification
    score_match = re.search(r'(\d+)\s*out of 10', scoring_output)
    if score_match:
        score = int(score_match.group(1))
    else:
        score = None

    # Extract the justification
    justification_start = scoring_output.lower().find('justification:')
    if justification_start != -1:
        justification = scoring_output[justification_start + len('justification:'):].strip()
    else:
        # If "justification:" not found, take the text after the score match
        score_end = score_match.end() if score_match else 0
        justification = scoring_output[score_end:].strip()

    return score, justification


#################################

def main():
    st.title("Resume Scoring Application")
    st.write("Upload resumes and provide a job description to score candidates based on their fit for the job.")

    # Job Description Input
    st.header("Job Description")
    job_description_text = st.text_area("Enter the job description here:", height=300)

    # Resume Upload
    st.header("Upload Resumes")
    uploaded_files = st.file_uploader("Choose PDF files", accept_multiple_files=True, type=['pdf'])

    if st.button("Process Resumes"):
        if not job_description_text:
            st.error("Please provide a job description.")
        elif not uploaded_files:
            st.error("Please upload at least one resume.")
        else:
            results = []

            # Process each uploaded resume
            for uploaded_file in uploaded_files:
                resume_text = read_pdf(uploaded_file)
                # Extract candidate name (assumed to be the first line)
                resume_lines = resume_text.strip().split('\n')
                if resume_lines:
                    candidate_name_line = resume_lines[0]
                    candidate_name = candidate_name_line.strip()
                else:
                    candidate_name = "Unknown"

                # Process the resume
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    score, justification = process_resume(resume_text, job_description_text)

                # Display the result
                st.subheader(f"Results for {candidate_name} ({uploaded_file.name})")
                st.write(f"**Score:** {score} out of 10")
                st.write(f"**Justification:** {justification}")

                # Append to results
                results.append({
                    'filename': uploaded_file.name,
                    'applicant_name': candidate_name,
                    'score': score,
                    'justification': justification
                })

            # Create a DataFrame
            df_results = pd.DataFrame(results)

            # Optionally, display the DataFrame
            st.header("Summary")
            st.dataframe(df_results)

            # Optionally, allow download of results
            csv = df_results.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name='resume_scoring_results.csv',
                mime='text/csv',
            )

if __name__ == "__main__":
    main()
