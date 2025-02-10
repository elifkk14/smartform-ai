
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from load_models import AIModelManager
from form_intent import FormIntentClassifier
from question_generator import QuestionGenerator
from user_behavior import UserBehaviorAnalyzer
from form_feedback import FormFeedbackGenerator
import traceback
import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# 📌 **Flask Başlat**
app = Flask(__name__)
CORS(app)

# 📌 **Tek Seferlik Model Yükleme**
AIModelManager.load_models()

# 📌 **Yüklenmiş modelleri çağır**
intent_classifier = FormIntentClassifier()
question_generator = QuestionGenerator()
user_behavior_analyzer = UserBehaviorAnalyzer()
feedback_generator = FormFeedbackGenerator()



@app.route('/analyze-ai', methods=['POST'])
def analyze_ai():
    try:
        data = request.json
        print(f"📌 **API'ye Gelen Veri:** {data}")

        if not data:
            return jsonify({"error": "Invalid JSON format."}), 400

        form_fields = data.get('questions', [])
        form_title = data.get('form_title', "").strip()
        form_questions = [f.get("name", "Unnamed Field") for f in form_fields if f.get("name")]
        user_input_question = data.get('user_question', "").strip()

        if not form_fields and not user_input_question:
            return jsonify({"error": "Form fields or user question are required."}), 400

        form_purpose = form_title if form_title else "Unknown"
        print(f"📌 **Form Başlığı Algılandı: {form_purpose}**")

        if form_purpose == "Unknown" and form_questions:
            try:
                form_purpose = intent_classifier.predict_intent(form_title, form_questions)
                print(f"📌 **AI Predicted Purpose:** {form_purpose}")  
            except Exception as e:
                print(f"🚨 AI intent detection error: {e}")

        # 📌 **Kullanıcı Davranış Verisini Simüle Et**
        user_data = [
            {
                "question_id": i,
                "question_text": form_questions[i],
                "time_spent": np.random.randint(5, 20),
                "skipped": np.random.choice([True, False], p=[0.2, 0.8])
            }
            for i in range(len(form_fields))
        ] if form_questions else []

        # 📌 **Kullanıcı Davranış Analizi**
        behavior_feedback = user_behavior_analyzer.analyze_behavior(user_data=user_data, form_questions=form_questions) if form_questions else []


        # 📌 **Form Alanlarını Hazırla**
        categorized_fields = [
            {
                "field_name": f.get("name", "Unnamed Field"),
                "type": f.get("type", "text"),
                "placeholder": f.get("placeholder", "")
            }
            for f in form_fields
        ] if form_fields else []

        # 📌 **AI Destekli Geri Bildirim**
        try:
            feedback = feedback_generator.generate_feedback(form_purpose, user_data, form_questions, categorized_fields) if form_questions else []
        except Exception as e:
            print(f"🚨 AI feedback generation error: {e}")
            feedback = ["There is not feedback generation"]

        # 📌 **Önerilen Soruların Optimizasyonu**
        try:
            existing_questions = {q.lower().strip() for q in form_questions} if form_questions else set()
            suggested_questions = question_generator.generate_questions(form_purpose, existing_questions, num_questions=3)

            filtered_questions = [
                q.split('. ', 1)[-1].strip() for q in suggested_questions 
                if q.lower().strip() not in existing_questions
            ] if suggested_questions and "failed" not in suggested_questions[0].lower() else ["No relevant AI-generated questions."]

        except Exception as e:
            print(f"🚨 AI question generation error: {e}")
            filtered_questions = ["AI question generation failed."]

        # 📌 **AI Destekli Soru Asistanı (Kullanıcı Girişi İçin)**
        try:
            if user_input_question:
                question_suggestions = question_generator.suggest_question_variants(user_input_question)
                if not isinstance(question_suggestions, list) or len(question_suggestions) == 0:
                    raise ValueError("Empty or invalid AI suggestions received.")
            else:
                question_suggestions = []
        except Exception as e:
            print(f"🚨 AI Question Assistant Error: {e}")
            question_suggestions = ["⚠ AI could not generate question suggestions."]


        # 📌 **Önerileri Optimize Et**
       # 📌 **Hata kontrolü ekle**
        if isinstance(behavior_feedback, list):
            optimized_suggestions = behavior_feedback + feedback  # Listeyse direkt birleştir
        elif isinstance(behavior_feedback, dict):
            optimized_suggestions = behavior_feedback.get("feedback", []) + feedback  # Dict ise "feedback" anahtarını al
        else:
            optimized_suggestions = feedback  # Beklenmeyen durum olursa sadece feedback kullan



        response_data = {
            "form_purpose": form_purpose,
            "categorized_fields": categorized_fields,
            "suggestions": optimized_suggestions,
            "suggested_questions": filtered_questions,
            "question_assistant": question_suggestions
        }

        print(f"📌 **API Yanıtı:** {response_data}")  
        return jsonify(response_data), 200

    except Exception as e:
        app.logger.error(f"🚨 Internal Server Error: {traceback.format_exc()}")
        return jsonify({"error": "Internal Server Error", "details": traceback.format_exc()}), 500

@app.route('/suggest-question', methods=['POST'])
def suggest_question():
    try:
        data = request.json
        user_question = data.get('question', "").strip()
        form_purpose = data.get('form_purpose', "General Inquiry Form").strip()  # Varsayılan olarak genel bir form tanımla

        if not user_question:
            return jsonify({"error": "No question provided."}), 400

        print(f"📌 Kullanıcının girdiği soru: {user_question}")
        print(f"📌 Form Başlığı: {form_purpose}")

        # AI'yı Kullanarak Formun Konusuna Uygun Yeni Bir Soru Üret
        suggestions = question_generator.generate_question_from_user_input(user_question, form_purpose)

        return jsonify({"suggested_variants": suggestions})

    except Exception as e:
        print(f"🚨 API Hatası: {e}")
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500



    
@app.route('/')
def home():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, port=5006)
