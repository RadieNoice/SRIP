import json
import logging
import time
import pandas as pd
import re
# CHANGE THIS LINE:
from langchain_openai import ChatOpenAI # Import ChatOpenAI instead of OllamaLLM
from langchain.prompts import PromptTemplate

# ------------------------
# 1. Setup
# ------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# --- Hyperparameters ---
MODEL_NAME = "deepseek/deepseek-r1-0528-qwen3-8b" # Make sure this model is loaded in LM Studio
TEMPERATURE = 0.3
MAX_TOKENS = 2000 # Set a reasonable max_tokens for the output

# CHANGE THIS BLOCK:
llm = ChatOpenAI(
    model=MODEL_NAME,
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS, # Note: num_predict becomes max_tokens for ChatOpenAI
    base_url="http://localhost:1234/v1", # LM Studio's OpenAI-compatible endpoint often requires /v1
    api_key="not-needed" # LM Studio does not require an API key
)


INPUT_PATH = r"C:\Users\dhili\Desktop\SRIP\SRIP_os_student_variations_first3.json"
JSON_OUTPUT_PATH = r"C:\Users\dhili\Desktop\SRIP\evaluation_deepseek-r1_temperature_03.json"
EXCEL_OUTPUT_PATH = r"C:\Users\dhili\Desktop\SRIP\evaluation_deepseek-r1_temperature_03.xlsx"

# ... (rest of your code remains the same)

# ------------------------
# 2. Prompt Template
# ------------------------
evaluation_prompt_template = PromptTemplate.from_template("""
You are a OS evaluator. Use the rubric and ideal answer to evaluate the student answer using the {strategy} strategy.

Question: {question}
Ideal Answer: {ideal_answer}
Rubric: {rubric}

Student Answer: {student_answer}

Respond in valid JSON like this:
{{
  "score": <integer from 0 to 5>,
  "evaluation_strategy": "{strategy}",
  "reasoning": "Explain why the score was given, referring to rubric items."
}}
""")

# ------------------------
# 3. Helpers
# ------------------------
def clean(text):
    # This function needs to be updated to extract the JSON block
    # from potentially noisy LLM output.
    # It looks for the first occurrence of '{' and the last occurrence of '}'
    # to extract the JSON string.
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        json_str = match.group(0)
        # Attempt to clean potential escaped underscores, though less critical now
        # if the LLM is good at generating valid JSON within the block.
        return json_str.replace('\\_', '_')
    else:
        # If no JSON block is found, return the original text or an empty string
        # This will likely lead to a JSONDecodeError, which your handler will catch.
        logging.warning(f"No JSON object found in LLM response: {text[:200]}...")
        return text # Return original text, json.loads will fail

def evaluate(question, ideal_answer, rubric, student_answer, strategy):
    prompt = evaluation_prompt_template.format(
        question=question,
        ideal_answer=ideal_answer,
        rubric=rubric,
        student_answer=student_answer,
        strategy=strategy
    )
    try:
        raw_llm_response = llm.invoke(prompt)
        raw_text = raw_llm_response.content if hasattr(raw_llm_response, 'content') else str(raw_llm_response)

        # Apply the improved clean function
        cleaned_json_string = clean(raw_text)

        parsed_json = json.loads(cleaned_json_string) # Use the cleaned string
        return parsed_json
    except json.JSONDecodeError as e:
        # Pass the original raw_text to the warning for better debugging
        logging.warning(f"⚠️ Failed to parse {strategy} evaluation for student answer: '{student_answer[:50]}...'. Error: {e}. Raw response (full): '{raw_text}'")
        return {
            "score": 0,
            "evaluation_strategy": strategy,
            "reasoning": f"Invalid JSON format returned. Error: {e}. Raw response: {raw_text}"
        }
    except Exception as e:
        raw_content_for_debug = raw_llm_response.content if 'raw_llm_response' in locals() and hasattr(raw_llm_response, 'content') else "N/A"
        logging.error(f"❌ An unexpected error occurred during evaluation for {strategy}: {e}. Raw LLM Content (if available): '{raw_content_for_debug[:100]}'")
        return {
            "score": 0,
            "evaluation_strategy": strategy,
            "reasoning": f"An unexpected error occurred: {e}"
        }


# ... (rest of your code)

# ------------------------
# 4. Export to Excel
# ------------------------
def export_to_excel(full_data, path):
    rows = []
    for item in full_data:
        for variant in item["student_variants"]:
            # Ensure the keys exist before accessing
            clar_eval = variant.get("clarification", {})
            targ_eval = variant.get("target_guided", {})

            rows.append({
                "question": item["question"],
                "student_answer": variant["student_answer"],
                "correct_answer": item["ideal_answer"],
                "rubric": json.dumps(item["rubric"]),
                "evaluation_strategy_clarification": clar_eval.get("evaluation_strategy", "N/A"),
                "score_clarification": clar_eval.get("score", -1), # Use -1 or another indicator for missing
                "reasoning_clarification": clar_eval.get("reasoning", "N/A"),
                "evaluation_strategy_target_guided": targ_eval.get("evaluation_strategy", "N/A"),
                "score_target_guided": targ_eval.get("score", -1),
                "reasoning_target_guided": targ_eval.get("reasoning", "N/A")
            })
    df = pd.DataFrame(rows)
    df.to_excel(path, index=False)
    logging.info(f"📄 Excel file saved to: {path}")



# ------------------------
# 5. Main Evaluation Only
# ------------------------
def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    final_data = []

    for qid, item in enumerate(dataset):
        question = item["question"]
        ideal_answer = item["ideal_answer"]
        rubric = item["rubric"]

        logging.info(f"🧠 Q{qid+1}: Evaluating {len(item['student_variants'])} variants for question: {question[:50]}...")
        student_variants_evaluated = []

        # Corrected: item["student_variants"] is a list of strings (student answers)
        for i, student_answer in enumerate(item["student_variants"]):
            logging.info(f"    📊 Evaluating variant {i+1} for student answer: {student_answer[:50]}...")

            clar = evaluate(question, ideal_answer, rubric, student_answer, "clarification")
            time.sleep(1) # Be mindful of rate limits or just to space out requests
            targ = evaluate(question, ideal_answer, rubric, student_answer, "target_guided")
            time.sleep(1)

            # Determine the "best" score based on a simple comparison
            best = clar if clar.get("score", 0) >= targ.get("score", 0) else targ

            student_variants_evaluated.append({
                "student_answer": student_answer,
                "clarification": clar,
                "target_guided": targ,
                "selected_evaluation": best
            })

        final_data.append({
            "question": question,
            "ideal_answer": ideal_answer,
            "rubric": rubric,
            "student_variants": student_variants_evaluated
        })

    with open(JSON_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)
    logging.info(f"✅ JSON file saved to: {JSON_OUTPUT_PATH}")

    export_to_excel(final_data, EXCEL_OUTPUT_PATH)

if __name__ == "__main__":
    main()