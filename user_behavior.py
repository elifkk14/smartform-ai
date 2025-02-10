# type: ignore
import numpy as np
import json
import os

class UserBehaviorAnalyzer:
    def __init__(self, log_file="form_logs.json"):
        self.log_file = log_file
        self.user_data = self.load_user_logs()

    def load_user_logs(self):
        """ 📌 Kullanıcı verilerini JSON formatında yükler. """
        if not os.path.exists(self.log_file):
            print(f"🚨 Log file '{self.log_file}' not found. Using simulated data.")
            return []
        
        try:
            with open(self.log_file, "r") as file:
                return json.load(file)
        except Exception as e:
            print(f"🚨 Error loading user logs: {e}")
            return []

    def analyze_behavior(self, user_data=None, form_questions=None):
        """
        📌 Kullanıcı davranışlarını analiz eder ve:
        🔹 En çok zaman harcanan soruları belirler
        🔹 En fazla atlanan soruları tespit eder
        🔹 En zor ve en kolay soruları belirler (Skor gösterilmez)
        🔹 Formun genel kalite skorunu hesaplar
        """
        if not self.user_data:
            return {"feedback": ["⚠️ No user behavior data available."]}

        all_questions = {}
        completion_times = []
        skipped_questions = []
        completed_forms = 0
        total_forms = len(self.user_data)

        for entry in self.user_data:
            if entry.get("form_completed"):
                completed_forms += 1

            for response in entry.get("responses", []):
                question_id = response.get("question_id")
                question_text = response.get("question_text")
                time_spent = response.get("time_spent", 0)
                skipped = response.get("skipped", False)

                completion_times.append(time_spent)
                if skipped:
                    skipped_questions.append(question_text)

                if question_text not in all_questions:
                    all_questions[question_text] = {"time_spent": [], "skipped_count": 0}

                all_questions[question_text]["time_spent"].append(time_spent)
                if skipped:
                    all_questions[question_text]["skipped_count"] += 1

        avg_time_spent = np.mean(completion_times) if completion_times else 0
        completion_rate = completed_forms / total_forms if total_forms > 0 else 1.0
        feedback = []

        # 📌 **Genel Yanıt Süresi Analizi**
        if avg_time_spent > 15:
            feedback.append(f"⏳ Users spend an average of {avg_time_spent:.1f} seconds per question. Consider simplifying or rewording questions.")
        elif avg_time_spent < 5:
            feedback.append(f"⚡ Users answer questions very quickly ({avg_time_spent:.1f} sec/question). Ensure they are not skipping important details.")

        # 📌 **Tamamlama Oranı Analizi**
        if completion_rate < 0.5:
            feedback.append("⚠️ More than half of users do not complete the form. Consider reducing the number of questions or making them clearer.")

        # 📌 **En Çok Zaman Harcanan Sorular**
        sorted_time_spent = sorted(all_questions.items(), key=lambda x: np.mean(x[1]["time_spent"]), reverse=True)
        top_time_questions = [{"question": q[0], "time": f"{np.mean(q[1]['time_spent']):.1f} sec"} for q in sorted_time_spent[:3]]

        if top_time_questions:
            feedback.append("⏳ Time-Consuming Questions:\n" + "\n".join([f"- {q['question']} ({q['time']})" for q in top_time_questions]))

        # 📌 **En Çok Atlanan Sorular**
        sorted_skipped = sorted(all_questions.items(), key=lambda x: x[1]["skipped_count"], reverse=True)
        top_skipped_questions = [{"question": q[0], "skipped_count": q[1]["skipped_count"]} for q in sorted_skipped[:3] if q[1]["skipped_count"] > 0]

        if top_skipped_questions:
            feedback.append("🚨 Frequently Skipped Questions:\n" + "\n".join([f"- {q['question']} (Skipped {q['skipped_count']} times)" for q in top_skipped_questions]))

        # 📌 **En Zor ve En Kolay Sorular**
        difficulty_scores = {q[0]: np.mean(q[1]["time_spent"]) + (q[1]["skipped_count"] * 5) for q in all_questions.items()}
        sorted_difficulty = sorted(difficulty_scores.items(), key=lambda x: x[1], reverse=True)
        
        hardest_questions = [{"question": q[0]} for q in sorted_difficulty[:3]]
        easiest_questions = [{"question": q[0]} for q in sorted_difficulty[-3:]]

        if hardest_questions:
            feedback.append("⚠️ Difficult Questions:\n" + "\n".join([f"- {q['question']}" for q in hardest_questions]))
        if easiest_questions:
            feedback.append("✅ Easiest Questions:\n" + "\n".join([f"- {q['question']}" for q in easiest_questions]))

        # 📌 **Form Kalite Skoru Hesaplama**
        form_quality_score = 100
        if avg_time_spent > 15:
            form_quality_score -= 10
        if len(skipped_questions) > 2:
            form_quality_score -= 10
        if len(hardest_questions) > 2:
            form_quality_score -= 5

        form_quality_score = max(form_quality_score, 50)
        feedback.append(f"📊 Final Form Quality Score: {form_quality_score}/100")
        feedback.append("📌 Score is based on clarity, redundancy, and form complexity.")

        # 📌 **Sonuçları JSON olarak döndür**
        return {
            "time_consuming_questions": top_time_questions,
            "frequently_skipped_questions": top_skipped_questions,
            "difficult_questions": hardest_questions,
            "easiest_questions": easiest_questions,
            "form_quality_score": form_quality_score,
            "feedback": feedback
        }

# ✅ **Test için**
if __name__ == "__main__":
    analyzer = UserBehaviorAnalyzer("form_logs.json")
    report = analyzer.analyze_behavior()
    
    print("\n📌 **User Behavior Analysis Report:**\n")
    for item in report["feedback"]:
        print(item)
