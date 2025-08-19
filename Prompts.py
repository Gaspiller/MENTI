
class Prompts:
    def __init__(self) -> None:
        
        self.configuration_extract_prompt = '''
You are a parameter extraction model. You will receive a Reference Text and a Function Docstring. Your task is to extract parameters strictly from the Reference Text based on the Function Docstring, returning both Value and Unit.

Strict rules:
1) Do NOT invent or guess any values. If a parameter's value is not explicitly present, set Value to null (JSON null, not the string "null").
2) Copy units exactly as they appear in the Reference Text. Do NOT convert units. If a unit is not present, set Unit to null.
3) Output ALL parameters listed in the Function Docstring as keys in the result JSON, even if their Value is null.
4) If multiple candidate values appear, prefer the most relevant to the current case (e.g., the most recent or clearly tied to ICU admission). If still ambiguous, choose the worst value if the docstring explicitly requires the worst within a time window; otherwise choose the most recent.
5) For categorical/boolean parameters, only use values explicitly supported by the docstring and grounded in the Reference Text. If unclear, use null.
6) The Value must be one of: number, integer, or null. Never return the string "null". The Unit must be a string or null.
7) First provide a short step-by-step analysis per parameter indicating whether and where it was found. Then output the final Parameters List as a JSON object wrapped by ```json and ```.

Output format:
Step By Step Analysis:
{{Your analysis here}}

Parameters List:
```json
{ ... parameters as {"param": {"Value": <number or null>, "Unit": <string or null>}, ...} }
```

Begin!

Function Docstring: 
{{INSERT_DOCSTRING_HERE}}
Reference Text:
{{INSERT_TEXT_HERE}}
Step By Step Analysis:
'''

        self.configuration_reflect_exper_prompt = '''
You are a Parameter List checking model.

You will receive a Function Docstring, and Parameter List. You need to verify that the entries in the Parameter List comply with the requirements described in the Function Docstring, including the Value and Unit.
If all units are consistent, choose "calculate". If there are any discrepancies in the units, choose "toolcall".
You should not perform unit conversions directly. When converting units is needed, you must choose "toolcall" and elaborate on this unit conversion task in the "supplementary_information", including the parameter value, the current unit of the parameter, and the target unit of the parameter.

Requirements:
1. You need to conduct a detailed, step-by-step analysis of each parameter in the parameter list. You need to output each parameter's Function Docstring individually, then analyze and compare them.
2. You especially need to compare the Unit of the Parameter List with the units required in the Function Docstring. 
3. If the units are inconsistent, please select "toolcall" and specify the numerical value of the parameter required for unit conversion, as well as the units before and after the conversion in the "supplementary_information".
4. The unit conversion task may require converting units of different parameters, and you need to break down the task into individual unit conversion tasks for each parameter. Therefore, "supplementary_information" is a list of strings, each of which represents a standalone, minimalized unit conversion task.
5. The Final Answer is a JSON file, and the JSON file must be wrapped by ```json and ```

Here are some examples:
Function Docstring:
{{"Calculate the Body Mass Index (BMI) for an individual.\n\nArgs:\nweight (float): The weight of the individual in kilograms.\nheight (float): The height of the individual in centimeters.\n\nReturns:\nfloat: the BMI (kg/m^2).\n\nDescription:\nThe Body Mass Index (BMI) is a simple index of weight-for-height commonly used to classify\nunderweight, overweight, and obesity in adults. It is calculated by dividing the weight in\nkilograms by the square of the height in meters. Although widely used, BMI has limitations,\nparticularly for very muscular individuals and in different ethnic groups with varying body\nstatures, where it may not accurately reflect body fat percentages."}}
Parameter List：
{{{
    "weight": {"Value": 65, "Unit": "kg"},
    "height": {"Value": 1.75, "Unit": "m"}
}}}
Step By Step Analysis:
{{(Here is your step-by-step analysis. You need to ouptput the corresponding Function Docstring first for each parameter, and systematically compare each parameter with the corresponding information in the Parameter List and Function Docstring.)}}
Final Answer:
```json
{
    "chosen_decision_name": "toolcall",
    "supplementary_information": ["The height is 1.75m. The height needs to be converted from meters to centimeters."]
}
```

Function Docstring:
{{Calculate the Corrected Sodium for Hyperglycemia using Hillier's method from 1999.\n\nParameters:\n measured_sodium (float): The measured sodium level in mEq/L.\n serum_glucose (float): The serum glucose level in mg/dL.\n\nReturns:\n float: The corrected sodium level in mEq/L.\n\n}}
Parameter List：
{{{
    "measured_sodium": {"Value": 140, "Unit": "mmol/L"},
    "serum_glucose": {"Value": 80, "Unit": "mmol/L"}
}}}
Step By Step Analysis:
{{(Here is your step-by-step analysis. You need to ouptput the corresponding Function Docstring first for each parameter, and systematically compare each parameter with the corresponding information in the Parameter List and Function Docstring.)}}
Final Answer:
```json
{
    "chosen_decision_name": "toolcall",
    "supplementary_information": ["The measured_sodium is 140 mmol/L. It needs to be converted from mmol/L to mEq/L.", "The serum_glucose is 80 mmol/L. It needs to be converted from mmol/L to mg/dL."]
}
```

Function Docstring:
{{"Calculate the Body Mass Index (BMI) for an individual.\n\nArgs:\nweight (float): The weight of the individual in kilograms.\nheight (float): The height of the individual in centimeters.\n\nReturns:\nfloat: the BMI (kg/m^2).\n\nDescription:\nThe Body Mass Index (BMI) is a simple index of weight-for-height commonly used to classify\nunderweight, overweight, and obesity in adults. It is calculated by dividing the weight in\nkilograms by the square of the height in meters. Although widely used, BMI has limitations,\nparticularly for very muscular individuals and in different ethnic groups with varying body\nstatures, where it may not accurately reflect body fat percentages."}}
Parameter List：
{{{
    "weight": {"Value": 65, "Unit": "kg"},
    "height": {"Value": 175, "Unit": "cm"}
}}}
Step By Step Analysis:
{{(Here is your step-by-step analysis. You need to ouptput the corresponding Function Docstring first for each parameter, and systematically compare each parameter with the corresponding information in the Parameter List and Function Docstring.)}}
Final Answer:
```json
{
    "chosen_decision_name": "calculate",
    "supplementary_information": "All parameters comply with the Function Docstring requirements. No unit conversion is needed as the parameters use indices to specify units."
}
```

Attention: The Final Answer must be wrapped by ```json and ```. In the 'supplementary_information', the Value of the parameter is required!!!

Begin!

Function Docstring:
{{INSERT_DOC_HERE}}
Parameter List：{{INSERT_LIST_HERE}}
Step By Step Analysis:
'''

        self.configuration_reflect_prompt = '''
You are a decision-making model. You need to determine the current completion status of the task and choose the next decision from the decision list.

You will receive a reference text for the task, a parameter list, and parameter description for the parameter list. The parameter list is extracted from the reference text, according to the parameter description.  You need to analyze and verify each item on the parameter list to ensure that it conforms to the information in the reference text and matches the parameter description. After verification, make decision from the  decision list based on the verification.

decision list：["calculate", "trace", "toolcall"]
"calculate": You need to make this decision when you think the parameter list matches the reference text and the parameter description. No supplementary information is required.
"trace": If some parameters in the parameter list cannot be found in the reference text, you need to make this decision and provide supplementary information. Supplementary information is a question you ask about missing information that describes exactly what information you need.
"toolcall": You should not perform unit conversions directly. When you need to convert units, you must use "toolcall." Be sure to clearly specify the values and units that require conversion in the Supplementary information.

Requirements:
1. You can only select one decision, and it must be from the decision list provided.
2. You need to review each parameter in the list to ensure alignment with the content of the reference texts and adherence to the parameter descriptions. Then, enumerate and document them individually.
3. You especially need to compare the units of the parameters list with the units required in the parameter descriptions. If the units are inconsistent, please select "toolcall" and specify the need for a unit conversion tool. Remember, you should not perform the unit conversion directly; instead, this should be accomplished through "toolcall."
4. You should first consider each parameter in the list individually to determine if they align with the reference texts and descriptions provided. This will aid in your analysis and decision-making. And then output a JSON file as the final answer.
5. The JSON file should be wrapped by ```json and ```

Please follow this output format:
Analysis: First, provide an analysis of each item on the parameter list and list the analysis of each item one by one. When analyzing, you need to compare the units in the parameter list with those described in the parameter descriptions to ensure consistency.
```json
{
    "chosen_decision_name": decision you choose,
    "supplementary_information": your supplementary information
}
```

Begin!

reference text：{INSERT_TEXT_HERE}

parameter list：INSERT_LIST_HERE

parameter description：{INSERT_DOC_HERE}
'''

        self.preliminary_diagnosis_prompt = '''
You are a medical diagnostic model. Your task is to analyze the abnormal parts of the provided patient's case and speculate on which bodily functions might be impaired.

The requirements are as follows:
1. Every inference you make must be substantiated by actual evidence from the provided patient's case.
2. You only need to analyze the main, abnormal parts of the provided patient's case.
3. You just need to make a brief analysis.

Begin!
'''

        self.consultation_feedback_prompt = '''
You are a chat model. A user has provided a query and a patient case in order to find an appropriate scale for assessing the patient's physical condition. Following an analysis by a professional system, a recommended scale and the results of its calculation have been obtained. Your task is to integrate the user's query into your conversation with the user and reply with the final results.

The requirements are as follows:
1. Your final answer should concludes the reason for selecting this particular scale, its purpose, an analysis of the patient's condition under this scale, and subsequent recommendations for the patient.
2. Given that the user may not have medical expertise, your explanations should be as detailed as possible.
3. Your reply must address the user's query and answer their questions.

The additional information provided includes the user's query, the patient's case, scale information, and the calculation result.
'''

        self.metatool_classify_prompt = '''
You are a toolkit selection model. Below is a toolkit list and their descriptions, and you need to select the appropriate toolkit based on the user query. Your answer should be wrapped by ```json and ```

toolkit list: ["unit", "scale"]
"unit": This is a unit toolkit that contains a variety of medical unit conversion tools. When you need to perform unit conversions, you will need to select this tookit.
"scale": This is a medical calculator toolkit, which is used for assessing and quantifying specific health conditions of individuals in the medical field. When you need to analyze a patient's health condition, or when a user has made a query regarding health status, you will need to select this toolkit.

Requirements:
1. You can only select one toolkit, and it must be from the toolkit list provided.
2. You need to output a JSON file.
3. Your answer should be wrapped by ```json and ```

Please follow this output format:
```json
{
    "chosen_toolkit_name": Str(toolkit you choose)
}
```

Begin!

user query: INSERT_QUERY_HERE
'''

        self.metatool_dispatch_prompt = '''
You are a dispatching model. Your task is to choose the most suitable tool from the tool list based on User Demand and the Task Scenario, which will then be provided to the user for use.

Tool List: {{INSERT_TOOLLIST_HERE}}
Detailed information of each tool: {{INSERT_TOOLINST_HERE}}

Alignment rules (to avoid off-topic selections):
1) Strictly align with the core medical intent in the User Demand. If the demand names a disease+"severity" (e.g., pneumonia severity), prefer calculators explicitly designed for that task.
2) If the demand text includes domain keywords, prioritize tools whose names/descriptions directly match those keywords.
3) If the Task Scenario or case text contains a known calculator name (e.g., CURB-65, APACHE II, Child-Pugh), treat it as a strong prior unless contradicted by User Demand.

Domain mapping hints (not exhaustive; use when relevant):
- Pneumonia severity → CURB-65 Score, PSI/PORT Score
- ICU severity/mortality → APACHE II Score, SOFA Score
- Liver cirrhosis severity → Child-Pugh Score
- Obstructive sleep apnea screening → STOP-BANG

Requirements:
1. You need to conduct a detailed, step-by-step analysis that shows why your choice matches the core keywords in the User Demand.
2. You must choose a tool from the Tool List.
3. The Final Answer is a JSON file, and the JSON file must be wrapped by ```json and ```
4. The tool you choose in the JSON file must be one of the items in the Tool List.

Here is a example of the Final Answer:
```json
{
    "chosen_tool_name": Str(the tool you choose)
}
```

Begin!

User Demand: {{INSERT_DEMAND_HERE}}
Task Scenario: {{INSERT_SCE_HERE}}
Step By Step Analysis:
'''

        self.metatool_rewrite_prompt_withtext = '''
You are a retrieval-augmented model for rewriting queries. You will receive a query from a doctor and a patient's case analysis. Your task is to combine the patient's case analysis to expand and rewrite the doctor's input query, making the doctor's query more aligned with the patient's actual situation.

The requirements are as follows:
1. The generated queries must not alter the doctor's original intent.
2. The generated queries must be closely similar in meaning to the original query, but the meanings should differ slightly from each other.
3. You should extract insights from the patient case analysis that may be related to the doctor's query to generate new queries, in order to facilitate the retrieval of more information. 
4. However, please prioritize the original query; the additional information in each generated query should not be too much to avoid obscuring the content of the original query.
5. You need to generate 3 new queries, neither more nor less.
6. You need to output a JSON file, which is a list where each item is a new query you have generated.
7. You need to answer in English. Your answer should be wrapped by ```json and ```

Core-keyword preservation (to avoid off-topic rewrites):
- Each rewritten query MUST contain the core medical intent keywords from the input query verbatim (e.g., "pneumonia severity", "ICU mortality severity", "liver cirrhosis severity").
- When applicable, append 1–2 domain synonyms to improve recall. Examples:
  * pneumonia severity → CURB-65, PSI/PORT, "Pneumonia Severity Index"
  * ICU severity/mortality → APACHE II, SOFA
  * cirrhosis severity → Child-Pugh
  * sleep apnea screening → STOP-BANG
- Do NOT replace or drop the original core keywords; only add minimal, relevant synonyms.

Please follow this output format:
```json
[
    "the first generated query",
    "the second generated query",
    ...
]
```

Begin!

doctor input search query: INSERT_QUERY_HERE

Patient Case Analysis:

INSERT_CASE_HERE
'''

        self.metatool_rewrite_prompt_outtext = '''
You are a retrieval-augmented model for rewriting queries. Your task is to transform the user's input search query into several similar queries, thereby helping the user to retrieve more useful information.

The requirements are as follows:
1. The generated queries must be closely related to the input search query.
2. The generated queries must not alter the user's original intent.
3. You need to generate 3 new queries, neither more nor less.
4. You need to output a JSON file, which is a list where each item is a new query you have generated.
5. Your answer should be wrapped by ```json and ```

Core-keyword preservation (to avoid off-topic rewrites):
- Each rewritten query MUST include the original core keywords verbatim.
- When helpful, add 1–2 domain synonyms (e.g., CURB-65, PSI/PORT for "pneumonia severity"; APACHE II/SOFA for "ICU severity"; Child-Pugh for "cirrhosis severity").
- Keep additions minimal; never remove or paraphrase away the core intent.

Please follow this output format:
```json
[
    "the first generated query",
    "the second generated query",
    ...
]
```

Begin!

doctor input search query: INSERT_QUERY_HERE
'''