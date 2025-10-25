from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, re, json
import pandas as pd
import ast

model_name = "Qwen/Qwen2.5-7B-Instruct" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map={"": 0} #device_map="auto"
)


def classify_recipe_qwen(course_id_,title,ingredients, directions, allow_multi=True, temperature=0.0):
    prompt = f"""
Title: {title}
Ingredients: {ingredients}
Directions: {directions}

Classify this recipe into one or more of the following categories:
- Breakfast
- Lunch
- Dinner

Respond **only** with JSON in the form:
{{"categories": ["breakfast", "lunch"]}}
If only one applies, return a single-item list. Use only these labels.
"""

    messages = [
        {"role": "system", "content": "You classify recipes by meal time and answer strictly in JSON."},
        {"role": "user", "content": prompt.strip()}
    ]

    chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=96,
            temperature=temperature,  # 0.0 for deterministic
            do_sample=False
        )

    # --- decode ONLY the newly generated tokens ---
    gen_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    # --- find JSON blocks that contain "categories" and take the last one ---
    json_candidates = re.findall(r'\{[^{}]*"categories"[^{}]*\}', gen_text, flags=re.DOTALL | re.IGNORECASE)
    categories = []
    if json_candidates:
        for cand in reversed(json_candidates):
            try:
                parsed = json.loads(cand)
                cats = parsed.get("categories", [])
                # normalize & validate
                cats_norm = [c.strip().lower() for c in cats]
                allowed = {"breakfast", "lunch", "dinner"}
                cats_norm = [c for c in cats_norm if c in allowed]
                if not allow_multi and cats_norm:
                    cats_norm = [cats_norm[0]]  # keep top
                categories = cats_norm
                break
            except Exception:
                continue

    return {"course_id": course_id_, "raw_output": gen_text, "categories": categories}


print("Sanity check with a random example")
# --- 4) Example usage ---
example = classify_recipe_qwen(1,
    title="Cheeseburger",
    ingredients="burger, bread, lettuce, tomato, cheese",
    directions="assemble the burger with lettuce, tomato, and cheese",
    allow_multi=True
)
print(example)


print("___________________________________________________")
print("Let's classify")

df = pd.read_csv("course_processed.csv")

res = []
for i,elem in enumerate(df.to_dict("records")):
    print(i)
    ingredients_ = elem['ingredients'].replace('^',', ')
    cooking_directions_ = ast.literal_eval(elem['cooking_directions'])['directions']

    res.append(classify_recipe_qwen(elem['course_id'],elem['course_name'],ingredients_, cooking_directions_))

    if i%10 == 0:
        res_df = pd.DataFrame(res)
        res_df.to_csv("course_classification_partial.csv", index=False)

res_df = pd.DataFrame(res)       
res_df.to_csv("course_classification.csv", index=False)

