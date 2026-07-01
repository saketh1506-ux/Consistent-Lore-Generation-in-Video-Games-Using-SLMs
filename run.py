"""
Tsushima NPC Dataset Generator
Generates ~300 fine-tuning examples for 5 NPCs x 3 villages using Gemini API.

USAGE:
    export GEMINI_API_KEY="your_key_here"
    python generate_dataset.py
"""

from google import genai
import json, os, random, time, re

# ---------- CONFIG ----------
API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL_NAME = "gemini-2.5-flash"
EXAMPLES_PER_PAIR = 20          # 5 NPCs x 3 villages x 20 = 300
SLEEP_BETWEEN_CALLS = 6         # ~10 req/min, safer margin under the 15 RPM free-tier cap
MAX_RETRIES = 5                 # retry on 429 rate-limit errors
RETRY_BACKOFF_SECONDS = 30      # wait time before retrying after a 429
LORE_DIR = "lore"
OUTPUT_FILE = "dataset_raw.jsonl"

NPC_FILES = ["yuna.json", "lord_shimura.json", "masako_adachi.json", "ryuzo.json", "khotun_khan.json"]

SYSTEM_PROMPT = "You are the Tsushima NPC AI Engine. Process the context and output a JSON containing internal thoughts, updated emotions, physical actions, and spoken dialogue."

# ---------- LOAD LORE ----------
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

characters = [load_json(os.path.join(LORE_DIR, f)) for f in NPC_FILES]
villages = load_json(os.path.join(LORE_DIR, "villages.json"))
events = load_json(os.path.join(LORE_DIR, "events.json"))

# Flatten events into a single list of (category, event) tuples
all_events = []
for category, evts in events.items():
    for e in evts:
        all_events.append((category, e))

# ---------- GENERATION PROMPT TEMPLATE ----------
def build_generation_prompt(character, village, event):
    char_json = json.dumps(character, indent=2)
    # pick a relevant relationship fact to use as memory log
    rel_keys = list(character["relationships"].keys())
    memory_subject = "Jin Sakai" if "Jin Sakai" in rel_keys else random.choice(rel_keys)
    memory_text = character["relationships"].get(memory_subject, character["known_facts"][0])

    return f"""You are generating ONE training example for an NPC AI engine for a game based on Ghost of Tsushima.

CHARACTER PROFILE (this NPC must stay strictly in-character):
{char_json}

LOCATION: {village['name']}
LOCATION_DESCRIPTION: {village['description']}

MEMORY_LOG (a relevant fact this NPC remembers about their relationship with Jin):
- {character['name']}'s relationship with Jin Sakai: {character['relationships'].get('Jin Sakai', memory_text)}

TRIGGER_EVENT: {event}
TRIGGER_ENTITY: Jin Sakai (the player character)

TASK:
Write a SHORT 1-2 sentence ENVIRONMENT description for this location (sensory detail, no fluff).
Then output ONLY valid JSON (no markdown, no commentary, no code fences) in EXACTLY this schema:

{{
  "internal_thought": "1-2 sentences max. First-person internal reasoning from this NPC.",
  "updated_emotion": {{"primary": "...", "intensity": 1-10, "secondary": "...", "intensity_2": 1-10}},
  "action": "1 sentence. Physical action grounded in the location.",
  "dialogue": "2-3 sentences max. What the NPC says out loud, in their speech_style."
}}

CRITICAL RULES:
- Keep everything SHORT and punchy. No long paragraphs.
- Emotion must be freshly reasoned per character and event. No defaulting to Fear/Determined every time.
- Each character must sound distinct per their speech_style.
- Stay consistent with known_facts and never_says.
- No placeholder text. Output must start with ENVIRONMENT: then JSON:

OUTPUT FORMAT (exactly):
ENVIRONMENT: <1-2 sentence environment description>
JSON: <your JSON object>
"""

def build_user_turn(character, village, environment, event):
    return (
        f"NPC: {character['name']}\n"
        f"Location: {village['name']}\n"
        f"Environment: {environment}\n"
        f"Memory Log:\n- {character['name']}'s relationship with Jin Sakai: {character['relationships'].get('Jin Sakai', '')}\n"
        f"Trigger Entity: Jin Sakai\n"
        f"Trigger Event: {event}"
    )

def clean_json(text):
    """Extract the JSON object from model output."""
    text = text.strip()
    text = re.sub(r"^```json", "", text)
    text = re.sub(r"^```", "", text)
    text = re.sub(r"```$", "", text)
    return text.strip()

def parse_response(raw_text):
    env_match = re.search(r"ENVIRONMENT:\s*(.+?)\nJSON:", raw_text, re.DOTALL)
    json_match = re.search(r"JSON:\s*(\{.*\})", raw_text, re.DOTALL)
    if not env_match or not json_match:
        raise ValueError("Could not parse ENVIRONMENT/JSON sections")
    environment = env_match.group(1).strip()
    json_str = clean_json(json_match.group(1))
    parsed = json.loads(json_str)
    return environment, parsed

# ---------- MAIN GENERATION LOOP ----------
def main():
    if not API_KEY:
        raise SystemExit("ERROR: set GEMINI_API_KEY environment variable first.")

    genai_client = genai.Client(api_key=API_KEY)

    dataset = []
    failures = []

    # Load already-completed events so we can skip them on resume
    completed = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as existing:
            for line in existing:
                try:
                    row = json.loads(line)
                    user_msg = row["messages"][1]["content"]
                    # extract NPC name, village, and event from user message
                    npc_line = [l for l in user_msg.split("\n") if l.startswith("NPC:")]
                    village_line = [l for l in user_msg.split("\n") if l.startswith("Location:")]
                    event_line = [l for l in user_msg.split("\n") if l.startswith("Trigger Event:")]
                    if npc_line and village_line and event_line:
                        key = (npc_line[0].replace("NPC:", "").strip(),
                               village_line[0].replace("Location:", "").strip(),
                               event_line[0].replace("Trigger Event:", "").strip())
                        completed.add(key)
                except Exception:
                    pass
        print(f"Resuming: {len(completed)} examples already done, skipping those.")

    out_f = open(OUTPUT_FILE, "a", encoding="utf-8")  # append mode, preserves prior progress

    for character in characters:
        for village_key, village in villages.items():
            # sample events for this NPC-village pair, spread across categories
            sampled = random.sample(all_events, min(EXAMPLES_PER_PAIR, len(all_events)))

            for category, event in sampled:
                # skip if already generated in a previous run
                skip_key = (character["name"], village["name"], event)
                if skip_key in completed:
                    print(f"SKIP| {character['name']:15s} | {village['name']:10s} | already done")
                    continue

                gen_prompt = build_generation_prompt(character, village, event)
                attempt = 0
                while attempt <= MAX_RETRIES:
                    try:
                        response = genai_client.models.generate_content(
                            model=MODEL_NAME, contents=gen_prompt
                        )
                        environment, parsed = parse_response(response.text)

                        user_content = build_user_turn(character, village, environment, event)
                        row = {
                            "messages": [
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": user_content},
                                {"role": "assistant", "content": json.dumps(parsed, ensure_ascii=False)}
                            ]
                        }
                        dataset.append(row)
                        out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        out_f.flush()
                        print(f"OK  | {character['name']:15s} | {village['name']:10s} | {category:18s} | {event[:50]}")
                        break  # success, exit retry loop

                    except Exception as e:
                        err_str = str(e)
                        is_retryable = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "503" in err_str or "UNAVAILABLE" in err_str
                        if is_retryable and attempt < MAX_RETRIES:
                            wait = RETRY_BACKOFF_SECONDS * (attempt + 1)
                            print(f"WAIT| {character['name']:15s} | rate limited, retrying in {wait}s (attempt {attempt+1}/{MAX_RETRIES})")
                            time.sleep(wait)
                            attempt += 1
                            continue
                        else:
                            failures.append({"character": character["name"], "village": village["name"], "event": event, "error": err_str})
                            print(f"FAIL| {character['name']:15s} | {village['name']:10s} | {err_str[:60]}")
                            break

                time.sleep(SLEEP_BETWEEN_CALLS)

    out_f.close()

    with open("generation_failures.json", "w") as f:
        json.dump(failures, f, indent=2)

    print(f"\nDone. {len(dataset)} examples written to {OUTPUT_FILE}")
    print(f"{len(failures)} failures logged to generation_failures.json")

if __name__ == "__main__":
    main()