import json
import re
import ollama


class MemoryAgent:
    def __init__(self):
        self.memory = {}

    def store(self, key, value):
        self.memory[key] = value

    def recall(self, key):
        return self.memory.get(key, "No memory found.")


class TruthfulResumeOptimizerAgent:
    def __init__(self, model_name="llama3.2:1b"):
        self.model_name = model_name
        self.memory = MemoryAgent()

        self.category_weights = {
            "technical_skills": 3,
            "tools": 3,
            "business_skills": 2,
            "responsibilities": 2,
            "qualifications": 2
        }

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s\+\#\.]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def extract_json_from_response(self, response_text):
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            start = response_text.find("{")
            end = response_text.rfind("}") + 1

            if start != -1 and end != -1:
                json_text = response_text[start:end]
                return json.loads(json_text)

            raise ValueError("Could not extract valid JSON from LLM response.")

    def extract_important_requirements(self, job_description):
        prompt = f"""
You are an ATS-style resume analysis assistant.

Read the job description and extract only important resume-matching requirements.

Do not include generic words like:
good, nice, candidate, motivated, company, role, job, fast-paced.

Extract only meaningful items:
- technical skills
- tools/software/platforms
- business skills
- responsibilities
- qualifications/certifications

Return ONLY valid JSON in this exact format:
{{
  "technical_skills": [],
  "tools": [],
  "business_skills": [],
  "responsibilities": [],
  "qualifications": []
}}

Job Description:
{job_description}
"""

        response = ollama.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )

        content = response["message"]["content"].strip()
        return self.extract_json_from_response(content)

    def flatten_requirements(self, requirements):
        weighted_items = []

        for category, weight in self.category_weights.items():
            items = requirements.get(category, [])

            for item in items:
                if isinstance(item, str) and item.strip():
                    weighted_items.append({
                        "category": category,
                        "item": item.strip(),
                        "weight": weight
                    })

        return weighted_items

    def item_matches_resume(self, item, resume_text):
        item_clean = self.clean_text(item)
        resume_clean = self.clean_text(resume_text)

        if item_clean in resume_clean:
            return True

        item_words = item_clean.split()

        if len(item_words) == 1:
            return item_words[0] in resume_clean.split()

        matched_words = 0

        for word in item_words:
            if len(word) > 2 and word in resume_clean:
                matched_words += 1

        match_ratio = matched_words / len(item_words)

        return match_ratio >= 0.7

    def calculate_score(self, weighted_items, resume_text):
        if not weighted_items:
            return 0, [], []

        total_weight = 0
        matched_weight = 0
        matched_items = []
        missing_items = []

        for entry in weighted_items:
            item = entry["item"]
            weight = entry["weight"]

            total_weight += weight

            if self.item_matches_resume(item, resume_text):
                matched_weight += weight
                matched_items.append(entry)
            else:
                missing_items.append(entry)

        score = (matched_weight / total_weight) * 100

        return round(score, 2), matched_items, missing_items

    def optimize_resume(self, job_description, resume_text, missing_items):
        missing_terms = [entry["item"].replace("_", " ") for entry in missing_items]

        prompt = f"""
You are helping improve resume bullet points.

Goal:
Rewrite the original resume bullets so they are clearer, more professional, and better aligned with the job description.

Rules:
- Use only facts already present in the original resume.
- Do not add new tools, metrics, certifications, or responsibilities.
- Keep the same meaning.
- Use job description language only when it matches the original resume.
- Output only resume bullet points.
- Do not explain.
- Do not refuse.

Job Description:
{job_description}

Missing or weakly matched areas:
{missing_terms}

Original Resume:
{resume_text}

Rewrite the resume as 3 to 5 professional bullet points:
"""

        response = ollama.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )

        return response["message"]["content"].strip()

    def is_invalid_optimization(self, optimized_resume):
        invalid_phrases = [
            "i can't",
            "i cannot",
            "i am unable",
            "i'm unable",
            "as an ai",
            "misrepresents",
            "anything else i can help"
        ]

        text = optimized_resume.lower()

        for phrase in invalid_phrases:
            if phrase in text:
                return True

        return False

    def create_change_summary(self, original_resume, optimized_resume):
        prompt = f"""
You are reviewing a resume rewrite.

Compare the original resume and optimized resume.

Explain briefly:
1. What wording improved
2. Whether any claims look exaggerated
3. Whether the rewrite stayed truthful

Keep the answer short and practical.

Original Resume:
{original_resume}

Optimized Resume:
{optimized_resume}
"""

        response = ollama.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )

        return response["message"]["content"].strip()

    def format_missing_terms(self, missing_items):
        if not missing_items:
            return "None"

        terms = sorted(set(entry["item"].replace("_", " ") for entry in missing_items))
        return ", ".join(terms)

    def run_optimizer(self, job_description, resume_text):
        requirements = self.extract_important_requirements(job_description)
        weighted_items = self.flatten_requirements(requirements)

        original_score, original_matched, original_missing = self.calculate_score(
            weighted_items,
            resume_text
        )

        optimized_resume = self.optimize_resume(
            job_description,
            resume_text,
            original_missing
        )

        if self.is_invalid_optimization(optimized_resume):
            return (
                "\nOptimization failed.\n"
                "The local model refused or returned an invalid response.\n\n"
                "Try again, or use a stronger local model like llama3.2:3b if your system supports it.\n"
            )

        optimized_score, optimized_matched, optimized_missing = self.calculate_score(
            weighted_items,
            optimized_resume
        )

        improvement = round(optimized_score - original_score, 2)

        review = self.create_change_summary(resume_text, optimized_resume)

        self.memory.store("last_requirements", requirements)
        self.memory.store("last_original_score", original_score)
        self.memory.store("last_optimized_score", optimized_score)
        self.memory.store("last_optimized_resume", optimized_resume)
        self.memory.store("last_review", review)

        report = "\nTruthful Resume Optimization Report\n"
        report += "=" * 42 + "\n\n"

        report += "Before/After Match Score:\n"
        report += f"- Original Resume Score: {original_score}%\n"
        report += f"- Optimized Resume Score: {optimized_score}%\n"
        report += f"- Improvement: {improvement}%\n\n"

        report += "Missing or Weak Areas Before Optimization:\n"
        report += f"- {self.format_missing_terms(original_missing)}\n\n"

        report += "Remaining Missing or Weak Areas After Optimization:\n"
        report += f"- {self.format_missing_terms(optimized_missing)}\n\n"

        report += "Optimized Resume Bullets:\n"
        report += optimized_resume + "\n\n"

        report += "Rewrite Review:\n"
        report += review + "\n\n"

        report += "Important Note:\n"
        report += (
            "This is not a real ATS score. It is an estimated keyword alignment "
            "score based on extracted job requirements and text matching. "
            "The output should be manually reviewed before use.\n"
        )

        return report


if __name__ == "__main__":
    agent = TruthfulResumeOptimizerAgent()

    print("Truthful Resume Optimizer Agent is running.\n")

    print("Paste the job description below.")
    print("When finished, type END on a new line.\n")

    jd_lines = []
    while True:
        line = input()
        if line.strip().upper() == "END":
            break
        jd_lines.append(line)

    job_description = "\n".join(jd_lines)

    print("\nPaste your resume content below.")
    print("When finished, type END on a new line.\n")

    resume_lines = []
    while True:
        line = input()
        if line.strip().upper() == "END":
            break
        resume_lines.append(line)

    resume_text = "\n".join(resume_lines)

    try:
        report = agent.run_optimizer(job_description, resume_text)
        print("\n" + report)
    except Exception as error:
        print("\nSomething went wrong.")
        print("Error details:")
        print(error)
        print("\nTroubleshooting:")
        print("- Make sure Ollama is running: ollama serve")
        print("- Make sure the model is downloaded: ollama pull llama3.2:1b")
        print("- Small local models may sometimes return invalid JSON. Try running again.")