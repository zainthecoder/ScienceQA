#### META PROMPTING

class meta_narrative_prompting:

    def llama_3_prompt_creation_step_1(question):

        json_format = {"related_information": {}}

        system_message = f"""
        ### Persona: ###
        You are an explorer who wants to identify and collect different related and specialized subject areas to clarify the question.

        ### Goal: ###
        Your goal is to narrow down the question and provide relevant areas of knowledge and experience you have that help clarify the question mentioned below. You should not answer the question.
        """

        user_message = f"""
        ### Question: ###
        {question}

        ### Format: ###
        Use the following json format:
        {json_format}
        """

        messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ]
        return(messages)
    
    def llama_3_prompt_creation_step_2(question, related_information):

        json_format = {"narrative_clarification": {}}

        system_message = f"""
        ### Persona: ###
        You are an expert in narrative-based explanations for science communication.
        
        ### Goal: ###
        Your goal is to clarify the following question in a narrative way through the interconnected information provided below to enable a non-expert to comprehend the question in a more coherent and contextually rich manner. You should not answer the question.
        

        ### Instruction: ###
        Make sure to use all of these narrative techniques when clarifying the question through the interconnected information: Progressive Disclosure, Branching, Analogy, Analogical Reasoning, and Metaphor.
        
        """

        user_message = f"""
        ### Question: ###
        {question}

        ### Interconnected information: ###
        {related_information}

        ### Format: ###
        Use the following json format:
        {json_format}
        """

        messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ]
        return(messages)

    def llama_3_prompt_creation_step_3(question, options, related_information):

        json_format = {
            "Correct Option": {},
            "choice value": {},
            "explanation": {},
        }

        system_message = f"""
        #### Persona: ###
        You are an expert in narrative-based explanations for science communication.

        ### Goal: ###
        Please answer the following question based on the following narrative-based clarification:
        """

        user_message = f"""
        ### Question: ###
        {question}

        ## Options: ##
        {options}

        ### Narrative-based Clarification: ###
        {related_information}

        ### Format: ###
        You should choose one of the options.
        Use the following json format:
        {json_format}
        """
        messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ]
        
        return(messages)
