# type: ignore
from load_models import AIModelManager
import torch

class FormIntentClassifier:
    def __init__(self):
        """ 📌 AI Modelini `load_models.py` içinden alır. Model zaten yüklenmişse tekrar yüklemez. """
        self.model, self.tokenizer = AIModelManager.get_intent_classifier()
        self.device = AIModelManager.get_device()

    def predict_intent(self, form_title, form_description, form_questions):
        """
        📌 **AI modeline formun amacını tahmin ettirir.**
        """
        print("📌 predict_intent çağrıldı")

        try:
            # **📌 Form başlığı, açıklaması ve sorular ile prompt oluştur**
            prompt = f"Analyze the following form and determine its purpose:\n\n"
            if form_title:
                prompt += f"Form Title: {form_title}\n"
            if form_description:
                prompt += f"Form Description: {form_description}\n"
            if form_questions:
                formatted_questions = "\n".join(form_questions)
                prompt += f"Form Questions:\n{formatted_questions}\n"
            prompt += "Purpose:"

            print(f"📌 **AI'ya Gönderilen Prompt:**\n{prompt}")

            # **📌 Modelden Yanıt Al**
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=50)
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # **📌 Çıktıyı Temizle ve Sadece İlk Cümleyi Al**
            predicted_purpose = generated_text.split("Purpose:")[-1].strip().split("\n")[0]

            print(f"📌 **AI Predicted Purpose:** {predicted_purpose}")
            return predicted_purpose

        except Exception as e:
            print(f"🚨 AI çağrısı sırasında hata oluştu: {e}")
            return "Unknown"

# **✅ Test için bağımsız kod**
if __name__ == "__main__":
    classifier = FormIntentClassifier()
    form_title = "Appointment Request Form"
    form_description = "This form is used to request an appointment with our office."
    form_questions = [
        "What is your name?",
        "What is your email address?",
        "What date would you like to request an appointment?"
    ]

    print(f"📌 **Testing predict_intent() with:** Title: '{form_title}', Description: '{form_description}', Questions: {form_questions}")
    
    predicted_category = classifier.predict_intent(form_title, form_description, form_questions)
    print(f"📌 **Formun Tahmini Amacı:** {predicted_category}")
