import subprocess
import re, json

def clean_json(raw_output):
    match = re.search(r"\{.*\}", raw_output, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            return {"raw_output": raw_output}
    return {"raw_output": raw_output}

def extract_intent(text):
    prompt = f"""
    You are an intent parser.

    Extract exactly thress fields from the input sentence:
    - action: the verb (only synonyms of get or find allowed)
    - object: the noun or class being acted upon (just the noun is enough, no need for adverbs to be included)
    - colour: colour that defines the object / noun

    Respond ONLY in JSON format, with keys: action, object, colour.
    Do not include explanations, code, or text outside JSON.
    Make sure the JSON is complete and properly closed.
    If a colour is not found, the output for colour shall be none.
    Respond only with compact JSON on one line.
    Parse the sentences clearly and only include the words that are present in the string.

    Sentence: "{text}" 
    JSON: 
    """
    # ollama for using llama / gemma
    result = subprocess.run(
        # ["ollama", "run", "llama3.2:1b"],
        ["ollama", "run", "llama3.2"],
        # ["ollama", "run", "gemma2:2b"],
        input=prompt,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )

    parsed = clean_json(result.stdout)
    return parsed

'''
ls = ["get a pen", "find a animal",  "search for a green apple", "grab a red ball", 'get me a violet snack']

for l in ls:
    print(extract_intent(l))
'''

