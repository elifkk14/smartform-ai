
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class QuestionGenerator:
    _instance = None  

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(QuestionGenerator, cls).__new__(cls)
            cls._instance._initialize_model()
        return cls._instance

    def _initialize_model(self):
        """ 📌 **AI modelini yalnızca bir kez yükler** """
        model_id = "google/gemma-2-2b-it"
        device = "mps" if torch.backends.mps.is_available() else "cpu" 
        
        print("🚀 **AI modeli yükleniyor, lütfen bekleyin...**")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.tokenizer.pad_token = self.tokenizer.eos_token  

            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                torch_dtype=torch.float16 if device != "cpu" else torch.float32,  
                device_map=device
            )

            self.device = device
            print(f"✅ **AI modeli başarıyla yüklendi! (Device: {device})**")

        except Exception as e:
            print(f"🚨 Model yüklenirken hata oluştu: {e}")
            self.model = None

    def generate_questions(self, form_category, existing_questions, num_questions=5):
        """
        📌 **Formun kategorisine göre AI destekli sorular üretir.**
        - Form başlığına uygun sorular üretir.
        - AI önerdiği soruların formdaki mevcut sorularla aynı olup olmadığını kontrol eder.
        """
        if self.model is None:
            print("🚨 **Model yüklenmedi! Soru üretilemiyor.**")
            return ["AI model is not loaded. Unable to generate questions."]

        prompt = f"""
        Generate {num_questions} unique, relevant questions specifically for a form titled "{form_category}".
        The questions should match the purpose of the form and provide useful input from the user.

        Example format:
        - If it's an "Appointment Request Form":
          1. What is the purpose of your appointment?
          2. What date and time are you available?
          3. Do you have any special requests for the appointment?
        
        - If it's a "Customer Feedback Form":
          1. How would you rate our service?
          2. What can we improve?
          3. Would you recommend us to others?

        Ensure that:
        - Each question is clearly written and numbered (1., 2., 3., etc.).
        - The response contains ONLY the questions (no additional text or explanations).
        """

        try:
            print(f"📌 **AI'ya Gönderilen Prompt:**\n{prompt}")  # 🔹 **Gönderilen prompt'u logla**

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=250)  # 🔹 **Yanıtı uzat**

            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            print(f"📌 **Raw AI Output:**\n{generated_text}")  # 🔹 **Yanıtı logla**

            # 📌 **Yanıtı temizleyelim**
            ai_generated_questions = []
            for line in generated_text.split("\n"):
                line = line.strip("-• ").strip()
                if line and line[0].isdigit():  
                    ai_generated_questions.append(line)

            if not ai_generated_questions:
                print("🚨 **AI'dan geçerli soru alınamadı!**")
                return ["AI failed to generate proper questions. Please try again."]

            # 📌 **Zaten formda var olan soruları kaldır**
            existing_questions_lower = {q.lower().strip() for q in existing_questions}
            filtered_questions = [q for q in ai_generated_questions if q.lower().strip() not in existing_questions_lower]

            # 📌 **Eğer AI uygun soru üretmezse, mantıklı varsayılan sorular öner**
            if not filtered_questions:
                print("🚨 **AI suggested questions overlap with existing ones. Using fallback questions.**")
                return self.get_fallback_questions(form_category)

            return filtered_questions[:num_questions]

        except Exception as e:
            print(f"🚨 AI Error Details: {str(e)}")
            return ["AI question generation failed due to an internal error."]

    def get_fallback_questions(self, form_category):
        """
        📌 **Eğer AI uygun soru üretemezse, form başlığına uygun varsayılan sorular döndür.**
        """
        fallback_questions = {
            "Appointment Request Form": [
                "1. What is the purpose of your appointment?",
                "2. What date and time are you available?",
                "3. Do you have any special requests for the appointment?"
            ],
            "Customer Feedback Form": [
                "1. How would you rate our service?",
                "2. What can we improve?",
                "3. Would you recommend us to others?"
            ],
            "Job Application Form": [
                "1. What is your highest level of education?",
                "2. Do you have relevant work experience for this position?",
                "3. Why do you want to work for our company?"
            ],
            "Survey Form": [
                "1. How frequently do you use our service?",
                "2. What features do you find most valuable?",
                "3. What improvements would you suggest?"
            ]
        }

        return fallback_questions.get(form_category, ["1. What information would you like to provide?", "2. What is your main concern?", "3. How can we assist you?"])
    
    def generate_questions(self, form_category, existing_questions, num_questions=5):
        """
        📌 **Formun kategorisine göre AI destekli sorular üretir.**
        - Form başlığına uygun sorular üretir.
        - AI önerdiği soruların formdaki mevcut sorularla aynı olup olmadığını kontrol eder.
        """
        if self.model is None:
            print("🚨 **Model yüklenmedi! Soru üretilemiyor.**")
            return ["AI model is not loaded. Unable to generate questions."]

        prompt = f"""
        Generate {num_questions} unique, relevant questions specifically for a form titled "{form_category}".
        The questions should match the purpose of the form and provide useful input from the user.

        Ensure that:
        - Each question is clearly written and numbered (1., 2., 3., etc.).
        - The response contains ONLY the questions (no additional text or explanations).
        """

        try:
            print(f"📌 **AI'ya Gönderilen Prompt:**\n{prompt}")

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=250)  

            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            print(f"📌 **Raw AI Output:**\n{generated_text}")

            # 📌 **Yanıtı temizleyelim**
            ai_generated_questions = []
            for line in generated_text.split("\n"):
                line = line.strip("-• ").strip()
                if line and line[0].isdigit():  
                    ai_generated_questions.append(line)

            if not ai_generated_questions:
                print("🚨 **AI'dan geçerli soru alınamadı!**")
                return ["AI failed to generate proper questions. Please try again."]

            # 📌 **Zaten formda var olan soruları kaldır**
            existing_questions_lower = {q.lower().strip() for q in existing_questions}
            filtered_questions = [q for q in ai_generated_questions if q.lower().strip() not in existing_questions_lower]

            if not filtered_questions:
                print("🚨 **AI suggested questions overlap with existing ones. Using fallback questions.**")
                return self.get_fallback_questions(form_category)

            return filtered_questions[:num_questions]

        except Exception as e:
            print(f"🚨 AI Error Details: {str(e)}")
            return ["AI question generation failed due to an internal error."]

    def generate_question_from_user_input(self, user_question, form_purpose):
        """
        📌 Kullanıcının girdisine göre form başlığına uygun AI destekli yeni bir soru üretir.
        """
        if self.model is None:
            print("🚨 **Model yüklenmedi! Soru üretilemiyor.**")
            return ["AI model is not loaded. Unable to generate questions."]

        # 📌 **Tutarlı Prompt Formatı**
        prompt = f"""
        The user has asked: "{user_question}".
        Generate a single, unique question that can be added to a form titled "{form_purpose}".
        Your response should **ONLY** contain the question and **nothing else**.
        - Do **NOT** provide explanations.
        - Do **NOT** include answer choices.
        - Do **NOT** use labels like "Question:", "Options:", or "Explanation:".
        - The output should be a **single, well-formed question** ending with a question mark (?).
        """

        try:
            print(f"📌 **AI'ya Gönderilen Prompt:**\n{prompt}")

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=100)

            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            print(f"📌 **Raw AI Output:**\n{generated_text}")

            # 📌 **Yanıttan yalnızca ilk soru cümlesini çek**
            lines = generated_text.split("\n")
            question = None

            for line in lines:
                line = line.strip("-• ").strip()
                if line.endswith("?"):  # İlk geçen soru cümlesini al
                    question = line
                    break  # İlk soruyu bulunca dur

            if not question:
                print("🚨 **AI uygun bir soru üretmedi. Varsayılan soru kullanılıyor.**")
                return ["Could you please provide more details on this topic?"]

            return [question]  # AI'nın ürettiği ilk soruyu döndür

        except Exception as e:
            print(f"🚨 AI Error Details: {str(e)}")
            return ["AI question generation failed due to an internal error."]


    def generate_question_from_prompt(self, prompt):
        """
        📌 Kullanıcının girdisine göre AI destekli yeni bir soru üretir.
        """
        if self.model is None:
            print("🚨 **Model yüklenmedi! Soru üretilemiyor.**")
            return ["AI model is not loaded. Unable to generate questions."]

        try:
            print(f"📌 **AI'ya Gönderilen Prompt:**\n{prompt}")

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=100)

            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            print(f"📌 **Raw AI Output:**\n{generated_text}")

            # 📌 **Yanıttan yalnızca ilk soru cümlesini çek**
            lines = generated_text.split("\n")
            question = None

            for line in lines:
                line = line.strip("-• ").strip()
                if line.endswith("?"):  # İlk geçen soru cümlesini al
                    question = line
                    break  # İlk soruyu bulunca dur

            if not question:
                print("🚨 **AI uygun bir soru üretmedi. Varsayılan soru kullanılıyor.**")
                return ["Could you please provide more details on this topic?"]

            return [question]  # AI'nın ürettiği ilk soruyu döndür

        except Exception as e:
            print(f"🚨 AI Error Details: {str(e)}")
            return ["AI question generation failed due to an internal error."]


# ✅ **Bağımsız Test için**
if __name__ == "__main__":
    question_gen = QuestionGenerator()

    print(f"📌 **Testing suggest_question_variants()**")
    user_question = "Where do you live?"
    improved_questions = question_gen.suggest_question_variants(user_question)

    print("🔹 **AI Destekli Alternatif Sorular:**")
    for i, q in enumerate(improved_questions, 1):
        print(f"{i}. {q}")