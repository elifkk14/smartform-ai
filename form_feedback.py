
import re
from sentence_transformers import SentenceTransformer, util
import logging

logger = logging.getLogger(__name__)

class FormFeedbackGenerator:
    def __init__(self):
        try:
            self.similarity_model = SentenceTransformer("all-MiniLM-L6-v2")
            self.max_question_length = 80  
            logger.info("✅ NLP Model Successfully Loaded.")
        except Exception as e:
            logger.error(f"🚨 Error loading NLP model: {e}")
            self.similarity_model = None

    def detect_long_questions(self, questions):
        """ 📌 **Uzun soruları tespit edip daha kısa versiyon önerir.** """
        suggestions = []
        for q in questions:
            if len(q) > self.max_question_length:
                suggestions.append(f"⚠ Question '{q[:50]}...' is too long. Consider making it more concise.")
        return suggestions

    def detect_redundant_questions(self, questions):
        """ 📌 **Benzer veya yinelenen soruları tespit eder.** """
        if not self.similarity_model or not questions:
            return ["⚠ Similarity analysis is unavailable due to a model loading error."]

        redundant_suggestions = []
        question_embeddings = self.similarity_model.encode(questions, convert_to_tensor=True)

        for i in range(len(questions)):
            for j in range(i + 1, len(questions)):
                similarity = util.pytorch_cos_sim(question_embeddings[i], question_embeddings[j]).item()
                if similarity > 0.85:
                    redundant_suggestions.append(f"⚠ '{questions[i]}' and '{questions[j]}' are too similar. Consider merging or removing one.")

        return redundant_suggestions

    def detect_missing_questions(self, categorized_fields):
        """ 📌 **Eksik olabilecek soruları belirler.** """
        missing_suggestions = []
        required_fields = {"email", "phone", "address", "zip"}

        existing_fields = {field["field_name"].lower() for field in categorized_fields}
        missing_fields = required_fields - existing_fields

        for field in missing_fields:
            missing_suggestions.append(f"⚠ Consider adding a '{field}' field to ensure completeness.")

        return missing_suggestions

    def analyze_question_flow(self, questions):
        """ 📌 **Soruların mantıksal sırasını değerlendirir.** """
        if not questions:
            return ["⚠ Question flow analysis unavailable."]

        flow_suggestions = []
        personal_info = {"name", "email", "phone", "address"}
        personal_questions = [q for q in questions if any(info in q.lower() for info in personal_info)]
        if not personal_questions:
            flow_suggestions.append("⚠ Consider adding basic personal information questions at the beginning.")

        return flow_suggestions

    def generate_feedback(self, questions, categorized_fields):
        """ 📌 **Form için AI destekli geri bildirim üretir.** """
        feedback = []

        if not questions:
            return ["⚠ No questions provided. Ensure the form contains valid fields."]

        if not categorized_fields:
            return ["⚠ No categorized fields found. Check the form structure."]

        # Run feedback checks
        feedback.extend(self.detect_long_questions(questions))
        feedback.extend(self.detect_redundant_questions(questions))
        feedback.extend(self.detect_missing_questions(categorized_fields))
        feedback.extend(self.analyze_question_flow(questions))

        # Debugging logs
        if not feedback:
            print("🚨 No feedback was generated. This might indicate missing or invalid input data.")
            return ["⚠ AI could not generate meaningful feedback. Check if the input data is correct."]

        return feedback
